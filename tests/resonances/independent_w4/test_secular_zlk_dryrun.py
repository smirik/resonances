"""Independent functional tests of stage 10 (secular and ZLK on the shared libration analysis) and of the dry-run.

Written from SPEC §5.7 and §9, impl/API.md, W4-TASKS stage 10, CLAUDE.md and the public docstrings only.
Every expected number below comes from a closed form or a hand calculation written in the comments, or from a
synthetic series whose truth is known by construction. No integration: each test runs in well under 2 s.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import ellipk, factorial, hyp2f1, poch

import resonances
from resonances.resonance.analysis.secular.analyse import analyse_secular
from resonances.resonance.analysis.secular.models import AngleTerm, LongPeriodSeries
from resonances.resonance.analysis.secular.preprocess import choose_cutoff, forced_waves
from resonances.resonance.analysis.settings import AnalysisSettings, DryRunSettings
from resonances.resonance.analysis.verdict import (
    Action,
    AngleStatus,
    AngleSubtype,
    Confidence,
    LibrationOrigin,
    ResonanceType,
    Status,
    status_code,
    verdict_from_status_code,
)
from resonances.resonance.analysis.zlk.analyse import analyse_zlk
from resonances.secular.free_elements import free_elements
from resonances.simulation import dry_run as dr

DEG = np.pi / 180.0
ARCSEC_PER_TURN = 1296000.0
RAD_PER_ARCSEC = 2 * np.pi / ARCSEC_PER_TURN
G5, G6, S6 = 4.25749319, 28.24552984, -26.34496354  # arcsec/yr, the package model's fundamentals (const.py)
FREQUENCIES = {'g5': G5, 'g6': G6, 's6': S6}


# --------------------------------------------------------------------------------------------------------------
# helpers: synthetic series with a known truth
# --------------------------------------------------------------------------------------------------------------


def _grid(span_yr, step_yr):
    return np.arange(0.0, span_yr + 0.5 * step_yr, step_yr)


def _secular(angle, t, e=None, key='g-g6', terms=None):
    """A secular angle g - g6 (single planetary mode: no forced wave, SPEC/preprocess docstring)."""
    n = len(t)
    if terms is None:
        terms = (AngleTerm('g', None, 1), AngleTerm('g', 6, -1))
    g_body = 40.0 * RAD_PER_ARCSEC  # the body's own perihelion rate, 40"/yr
    return LongPeriodSeries(
        resonance_type=ResonanceType.SECULAR,
        key=key,
        name='synthetic',
        t_yr=t,
        angle=np.asarray(angle, dtype=float),
        a=np.full(n, 2.5),
        e=np.full(n, 0.1) if e is None else e,
        inc=np.full(n, 5 * DEG),
        varpi=g_body * t,
        Omega=-g_body * t,
        terms=terms,
        planet_frequencies=dict(FREQUENCIES),
    )


def _zlk_series(t, e, inc, varpi, Omega):
    omega = np.unwrap(varpi - Omega)
    return LongPeriodSeries(
        resonance_type=ResonanceType.LIDOV_KOZAI,
        key='LK',
        name='synthetic',
        t_yr=t,
        angle=omega,
        a=np.full(len(t), 2.5),
        e=e,
        inc=inc,
        varpi=varpi,
        Omega=Omega,
    )


def _from_vectors(z_e, z_i):
    """Osculating e, inc, varpi, Omega of the vectors z_e = e exp(i varpi), z_i = sin(i/2) exp(i Omega)."""
    e = np.abs(z_e)
    inc = 2 * np.arcsin(np.abs(z_i))
    varpi = np.unwrap(np.angle(z_e))
    Omega = np.unwrap(np.angle(z_i))
    return e, inc, varpi, Omega


SETTINGS = AnalysisSettings()
GATE_ON = SimpleNamespace(free_gate_enabled=True, free_gate_min_cycles=3.0)


# --------------------------------------------------------------------------------------------------------------
# SPEC §5.7: the integer status <-> the shared verdict (secular and ZLK)
# --------------------------------------------------------------------------------------------------------------

# (code, status, angle_status, reason) straight from the table of SPEC §5.7.
TABLE_5_7 = [
    (2, Status.RESONANT, AngleStatus.LIBRATION, 'dynamic_libration'),
    (1, Status.TRANSIENT, AngleStatus.EPISODIC, 'dynamic_episode'),
    (0, Status.NON_RESONANT, AngleStatus.CIRCULATION, 'circulation'),
    (-2, Status.NON_RESONANT, AngleStatus.LIBRATION, 'kinematic_libration'),
    (-3, Status.NEAR_SEPARATRIX, AngleStatus.NEAR_SEPARATRIX, 'separatrix_zone'),
    (-4, Status.UNCERTAIN, AngleStatus.INSUFFICIENT_TIME, 'baseline_needed'),
    (-5, Status.NEAR_SEPARATRIX, AngleStatus.NEAR_SEPARATRIX, 'separatrix_zone'),
    (-6, Status.UNCERTAIN, AngleStatus.DRIFT, 'baseline_needed'),
    (-99, Status.INVALID, AngleStatus.UNDEFINED, 'unphysical'),
]


@pytest.mark.parametrize('rtype', [ResonanceType.SECULAR, ResonanceType.LIDOV_KOZAI])
@pytest.mark.parametrize('code,status,angle_status,reason', TABLE_5_7)
def test_status_code_table_5_7_round_trip(rtype, code, status, angle_status, reason):
    v = verdict_from_status_code(code, rtype, confidence='medium')
    assert v.status == status
    assert v.angle_status == angle_status
    assert str(v.reason) == reason
    # exact inverse: the old integer is recovered (SPEC §5.7: status_code = legacy.status)
    assert v.status_code == code
    assert status_code(rtype, v.status, v.angle_status, v.reason, v.flags) == code


def test_minus_nine_reason_depends_on_type():
    # SPEC §5.7: -9 is gate_unavailable for ZLK, unusable_series for secular; both uncertain/undefined, action extend
    zlk = verdict_from_status_code(-9, ResonanceType.LIDOV_KOZAI)
    sec = verdict_from_status_code(-9, ResonanceType.SECULAR)
    for v in (zlk, sec):
        assert v.status == Status.UNCERTAIN
        assert v.angle_status == AngleStatus.UNDEFINED
        assert v.action == Action.EXTEND
        assert v.status_code == -9
    assert str(zlk.reason) == 'gate_unavailable'
    assert str(sec.reason) == 'unusable_series'


def test_kinematic_minus_two_and_unavailable_minus_nine_never_merge():
    # CLAUDE.md: -2 is a measurement (kinematic libration), -9 the absence of one; they must stay apart.
    kin = verdict_from_status_code(-2, ResonanceType.LIDOV_KOZAI)
    una = verdict_from_status_code(-9, ResonanceType.LIDOV_KOZAI)
    assert kin.libration_origin == LibrationOrigin.KINEMATIC
    assert kin.action == Action.NONE
    assert (kin.status, str(kin.reason)) != (una.status, str(una.reason))
    assert kin.status_code != una.status_code


def test_probable_near_separatrix_is_low_with_flag_and_review():
    # SPEC §5.7: -5 -> near_separatrix, confidence low, action review; told apart from -3 only by the flag
    v = verdict_from_status_code(-5, ResonanceType.SECULAR, confidence='high')
    assert v.confidence == Confidence.LOW
    assert 'probable_near_separatrix' in v.flags
    assert v.action == Action.REVIEW
    assert verdict_from_status_code(-3, ResonanceType.SECULAR, confidence='medium').action == Action.SPOT_CHECK


def test_legacy_confidence_is_uncalibrated_and_never_certain():
    # SPEC §5.7: `certain` is not given to these types before their own calibration; the flag says uncalibrated
    v = verdict_from_status_code(2, ResonanceType.SECULAR, confidence='high')
    assert v.confidence == Confidence.HIGH
    assert 'confidence_uncalibrated' in v.flags
    w = verdict_from_status_code(2, ResonanceType.LIDOV_KOZAI, confidence='certain')
    assert w.confidence != Confidence.CERTAIN


def test_minus_one_is_refused():
    # API: -1 is produced by no classifier and is refused
    with pytest.raises(ValueError):
        verdict_from_status_code(-1, ResonanceType.SECULAR)


# --------------------------------------------------------------------------------------------------------------
# secular/preprocess: the cutoff rule and the forced waves (closed forms)
# --------------------------------------------------------------------------------------------------------------


def test_forced_waves_of_4305_two_mode_angle():
    # 4305 g - 3g6 + 2g5 with g = 76.2"/yr (preprocess docstring): P_j = 1296000 / |g - g_j| yr
    #   g6: 1296000 / (76.2 - 28.24553) = 1296000 / 47.95447 = 27026 yr
    #   g5: 1296000 / (76.2 - 4.25749) = 1296000 / 71.94251 = 18014 yr
    terms = (AngleTerm('g', None, 1), AngleTerm('g', 6, -3), AngleTerm('g', 5, 2))
    waves = forced_waves(terms, 76.2, float('nan'), FREQUENCIES)
    assert sorted(waves.values()) == pytest.approx([1296000 / (76.2 - G5), 1296000 / (76.2 - G6)], rel=1e-9)
    assert sorted(waves.values()) == pytest.approx([18014, 27026], abs=1)


def test_forced_waves_of_mixed_g_s_angle():
    # z1 = g - g6 + s - s6 with g = 40, s = -45 "/yr: g-mode wave at g - g6, s-mode wave at s - s6
    #   1296000 / (40 - 28.24553) = 110 257 yr;  1296000 / |-45 + 26.34496| = 1296000 / 18.65504 = 69 472 yr
    terms = (AngleTerm('g', None, 1), AngleTerm('g', 6, -1), AngleTerm('s', None, 1), AngleTerm('s', 6, -1))
    waves = forced_waves(terms, 40.0, -45.0, FREQUENCIES)
    assert sorted(waves.values()) == pytest.approx(sorted([1296000 / (40 - G6), 1296000 / abs(-45 - S6)]), rel=1e-9)


def test_single_mode_angle_has_no_forced_wave():
    # nu6 = g - g6: the forced term at g6 is the resonance itself, so no wave (preprocess docstring)
    terms = (AngleTerm('g', None, 1), AngleTerm('g', 6, -1))
    assert forced_waves(terms, 40.0, -45.0, FREQUENCIES) == {}


def test_secular_cutoff_closed_form():
    # P_c = max(F_GI x 937, 2 x P_w, P_min / FILTER_MIN_LIBRATION_CUTOFFS)
    #     = max(2 x 937 = 1874, none, 2e5 / 2 = 1e5) = 1e5 yr  (libration bound); the step does not enter
    t = _grid(1.0e7, 1000.0)
    cut = choose_cutoff(_secular(np.zeros_like(t), t), SETTINGS)
    assert cut.cutoff_yr == pytest.approx(1.0e5, rel=1e-9) and cut.binding == 'shortest_libration'
    assert cut.gi_bound_yr == pytest.approx(2 * 937.0, rel=1e-9)
    assert cut.libration_bound_yr == pytest.approx(1.0e5, rel=1e-9)


def test_secular_cutoff_measures_body_g_and_forced_bound():
    # 4305-like angle on a body whose varpi turns at exactly 76.2"/yr: forced bound 2 x 27026 = 54052 yr,
    # still below the libration bound 1e5 yr, which binds.
    t = _grid(1.0e7, 1000.0)
    terms = (AngleTerm('g', None, 1), AngleTerm('g', 6, -3), AngleTerm('g', 5, 2))
    series = _secular(np.zeros_like(t), t, terms=terms, key='g-3g6+2g5')
    series.varpi = 76.2 * RAD_PER_ARCSEC * t
    cut = choose_cutoff(series, SETTINGS)
    assert cut.body_g_arcsec_yr == pytest.approx(76.2, abs=0.2)
    assert cut.forced_bound_yr == pytest.approx(2 * 1296000 / (76.2 - G6), rel=0.01)
    assert cut.cutoff_yr == pytest.approx(1.0e5, rel=1e-9)


@pytest.mark.parametrize(
    'step,expected',
    [
        # ZLK: max(2 x 937 = 1874, 2e3 / 2 = 1e3) = 1874 yr (great inequality binds)
        (10.0, 1874.0),
        # the same at a 500-yr step: the output step does not enter the cutoff
        (500.0, 1874.0),
    ],
)
def test_zlk_cutoff_closed_form(step, expected):
    t = _grid(200 * step * 50, step)
    n = len(t)
    series = _zlk_series(t, np.full(n, 0.2), np.full(n, 0.5), np.zeros(n), np.zeros(n))
    assert choose_cutoff(series, SETTINGS).cutoff_yr == pytest.approx(expected, rel=1e-6)


# --------------------------------------------------------------------------------------------------------------
# analyse_secular on synthetic angles (span 10 Myr, step 1 kyr, P_c = 100 kyr)
# --------------------------------------------------------------------------------------------------------------

T_SEC = 1.0e7
T_GRID = _grid(T_SEC, 1000.0)


def _sine(period, amplitude_deg=60.0, phase=0.0):
    return amplitude_deg * DEG * np.sin(2 * np.pi * T_GRID / period + phase)


def test_secular_stationary_libration_is_resonant():
    # 10 cycles of 1 Myr at 60 deg around a fixed centre: >= 3 turning points, centre fixed -> resonant (code 2)
    res = analyse_secular(_secular(_sine(1.0e6), T_GRID), SETTINGS)
    v = res.verdict
    assert v.status == Status.RESONANT
    assert v.angle_status == AngleStatus.LIBRATION
    assert str(v.reason) == 'dynamic_libration'
    assert v.status_code == 2
    assert res.libration_period_yr == pytest.approx(1.0e6, rel=0.05)
    # every margin full and no cap: certain, shown without a label minimum since W8 (author's decision 7, 2026-10-16)
    assert v.confidence == Confidence.CERTAIN


def test_secular_circulation_is_non_resonant():
    # 5 full turns in 10 Myr, monotone: circulation (code 0)
    res = analyse_secular(_secular(2 * np.pi * T_GRID / 2.0e6, T_GRID), SETTINGS)
    assert res.verdict.status == Status.NON_RESONANT
    assert res.verdict.angle_status == AngleStatus.CIRCULATION
    assert res.verdict.status_code == 0


@pytest.mark.parametrize(
    'travel_deg,subtype',
    [
        # SPEC §4.3.6: no turning point; range of sigma_f >= pi is an arc, < pi a stall
        (270.0, AngleSubtype.ARC),
        (90.0, AngleSubtype.STALL),
    ],
)
def test_secular_monotone_below_a_turn_needs_a_longer_baseline(travel_deg, subtype):
    # monotone travel below a full turn: no turning point -> insufficient_time, baseline_needed, extend, code -4
    res = analyse_secular(_secular(travel_deg * DEG * T_GRID / T_SEC, T_GRID), SETTINGS)
    v = res.verdict
    assert v.status == Status.UNCERTAIN
    assert v.angle_status == AngleStatus.INSUFFICIENT_TIME
    assert v.angle_subtype == subtype
    assert str(v.reason) == 'baseline_needed'
    assert v.action == Action.EXTEND
    assert v.status_code == -4


def test_secular_two_turning_points_are_not_a_libration():
    # 60 sin(2 pi t / 12 Myr): extrema at 3 and 9 Myr only (2 turning points, < LIBRATION_MIN_TURNING_POINTS = 3).
    # The centre check (CLAUDE.md, W4 stage 10): a bounded angle that turns back fewer times than a cycle is
    # uncertain, a longer integration would tell.
    res = analyse_secular(_secular(_sine(1.2e7), T_GRID), SETTINGS)
    v = res.verdict
    assert v.status == Status.UNCERTAIN
    assert v.angle_status in (AngleStatus.INSUFFICIENT_TIME, AngleStatus.DRIFT)
    assert str(v.reason) == 'baseline_needed'
    assert v.action == Action.EXTEND


def test_secular_three_turning_points_are_one_cycle_enough():
    # 60 sin(2 pi t / 6 Myr): extrema at 1.5, 4.5, 7.5 Myr = exactly 3 turning points; the first is 1.5 Myr and
    # the last 2.5 Myr from the edges, both within one period (6 Myr): a covering episode -> libration.
    res = analyse_secular(_secular(_sine(6.0e6), T_GRID), SETTINGS)
    assert res.verdict.status == Status.RESONANT
    assert res.verdict.angle_status == AngleStatus.LIBRATION
    assert res.verdict.confidence not in (Confidence.CERTAIN,)


def test_secular_moving_centre_is_not_resonant():
    # 30-deg libration of 1 Myr on a centre that moves monotonically by 330 deg in 10 Myr (less than a turn, never
    # turns back): the centre check makes such a record uncertain (drift, -6), never resonant.
    angle = _sine(1.0e6, amplitude_deg=30.0) + 330.0 * DEG * T_GRID / T_SEC
    v = analyse_secular(_secular(angle, T_GRID), SETTINGS).verdict
    assert v.status != Status.RESONANT
    assert v.status == Status.UNCERTAIN
    assert v.angle_status == AngleStatus.DRIFT
    assert v.status_code == -6


def test_secular_libration_then_circulation_is_transient():
    # 0-5 Myr: 60 sin(2 pi t / 1 Myr) (5 cycles, ends at sigma = 0); 5-10 Myr: 5 turns of circulation.
    angle = np.where(
        T_GRID < 5.0e6,
        60 * DEG * np.sin(2 * np.pi * T_GRID / 1.0e6),
        2 * np.pi * (T_GRID - 5.0e6) / 1.0e6,
    )
    v = analyse_secular(_secular(angle, T_GRID), SETTINGS).verdict
    assert v.status == Status.TRANSIENT
    assert v.angle_status == AngleStatus.EPISODIC
    assert str(v.reason) == 'dynamic_episode'
    assert v.status_code == 1


def test_secular_unphysical_orbit_is_invalid():
    # e > 1.3 is unphysical (is_unphysical_orbit): invalid / undefined, code -99, whatever the angle does
    e = np.full(len(T_GRID), 0.1)
    e[len(e) // 2] = 1.5
    v = analyse_secular(_secular(_sine(1.0e6), T_GRID, e=e), SETTINGS).verdict
    assert v.status == Status.INVALID
    assert v.angle_status == AngleStatus.UNDEFINED
    assert v.status_code == -99


def test_secular_nan_angle_is_unusable():
    angle = _sine(1.0e6)
    angle[100] = np.nan
    v = analyse_secular(_secular(angle, T_GRID), SETTINGS).verdict
    assert v.status == Status.UNCERTAIN
    assert v.angle_status == AngleStatus.UNDEFINED
    assert str(v.reason) == 'unusable_series'
    assert v.status_code == -9


def _integrate(rate, t):
    """Cumulative trapezoid of `rate` on `t`."""
    out = np.zeros_like(rate)
    out[1:] = np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))
    return out


def test_secular_e_response_measures_a_driven_e_and_changes_no_status():
    # Lagrange: a resonant secular g-term drives e_dot = K sin(sigma). Build e with exactly that, K = 2e-8 /yr
    # = 2e-5 per kyr: the fit e_dot = A sin + B cos + C must find R^2 ~ 1, amplitude 2e-5 /kyr, phase 0.
    sigma = _sine(1.0e6)
    k_per_yr = 2.0e-8
    e_driven = 0.1 + _integrate(k_per_yr * np.sin(sigma), T_GRID)
    driven = analyse_secular(_secular(sigma, T_GRID, e=e_driven), SETTINGS)
    r = driven.response
    assert 'e' in r.variables
    assert r.r2_e > 0.9
    assert r.amplitude_e_per_kyr == pytest.approx(k_per_yr * 1000, rel=0.05)
    assert abs((r.phase_e_deg + 180) % 360 - 180) < 10
    assert r.check == 'pass'

    # e that ignores sigma (a slow 3.7-Myr wave): no response; the status is unchanged (a column only, Q11)
    e_free = 0.1 + 0.01 * np.sin(2 * np.pi * T_GRID / 3.7e6)
    blind = analyse_secular(_secular(sigma, T_GRID, e=e_free), SETTINGS)
    assert blind.response.r2_e < 0.5
    assert blind.response.check != 'pass'
    assert blind.verdict.status == driven.verdict.status == Status.RESONANT
    assert blind.verdict.angle_status == driven.verdict.angle_status


# --------------------------------------------------------------------------------------------------------------
# analyse_zlk: the free-omega gate on synthetic vectors (span 5 Myr, step 500 yr)
# --------------------------------------------------------------------------------------------------------------

T_ZLK = _grid(5.0e6, 500.0)


def _kinematic_vectors():
    """Forced constant vectors dominate the free circles: z_e = 0.1 + 0.02 exp(i 50"/yr t),
    z_i = 0.1 + 0.02 exp(-i 50"/yr t). The osculating varpi and Omega each stay within asin(0.2) = 11.5 deg of 0,
    so omega = varpi - Omega is bounded (|omega| <= 23 deg, period 1296000/50 = 25.9 kyr, 193 cycles); the free
    omega = 50 t - (-50 t) winds at 100"/yr, 386 turns, at a rate far from every g_j, s_j and g_j - s_k
    (largest |g_j - s_k| = 28.25 + 26.34 = 54.6"/yr). Truth: a kinematic libration."""
    w = 50.0 * RAD_PER_ARCSEC * T_ZLK
    return 0.1 + 0.02 * np.exp(1j * w), 0.1 + 0.02 * np.exp(-1j * w)


def _genuine_vectors():
    """No forced part; the free node turns at -50"/yr and omega librates about 90 deg (30 deg, 200 kyr, 25 cycles):
    varpi = Omega + omega. Truth: a ZLK libration on the body's own, non-planetary rates."""
    Omega = -50.0 * RAD_PER_ARCSEC * T_ZLK
    omega = 90 * DEG + 30 * DEG * np.sin(2 * np.pi * T_ZLK / 2.0e5)
    return 0.3 * np.exp(1j * (Omega + omega)), 0.2 * np.exp(1j * Omega)


def _zlk(z_e, z_i, with_free=True, gate_config=GATE_ON):
    e, inc, varpi, Omega = _from_vectors(z_e, z_i)
    series = _zlk_series(T_ZLK, e, inc, varpi, Omega)
    fe = free_elements(T_ZLK, e, inc, Omega, varpi - Omega, sampling_years=500.0) if with_free else None
    return analyse_zlk(series, SETTINGS, None, fe, gate_config)


def test_zlk_kinematic_libration_is_demoted_to_minus_two():
    res = _zlk(*_kinematic_vectors())
    v = res.verdict
    assert v.angle_status == AngleStatus.LIBRATION  # the osculating omega does librate
    assert v.status == Status.NON_RESONANT
    assert str(v.reason) == 'kinematic_libration'
    assert v.libration_origin == LibrationOrigin.KINEMATIC
    assert v.status_code == -2
    assert res.gate.outcome == 'kinematic'


def test_zlk_genuine_libration_is_confirmed():
    res = _zlk(*_genuine_vectors())
    v = res.verdict
    assert res.gate.outcome == 'confirmed'
    assert v.status == Status.RESONANT
    assert v.angle_status == AngleStatus.LIBRATION
    assert v.status_code == 2
    assert res.libration_period_yr == pytest.approx(2.0e5, rel=0.05)
    # n_cycles = baseline / P_cyc = 5 Myr / 200 kyr = 25
    assert res.gate.n_cycles == pytest.approx(25.0, rel=0.1)


def test_zlk_without_free_elements_is_minus_nine_not_minus_two():
    # The same kinematic omega without free elements: no verdict is possible -> uncertain, gate_unavailable (-9),
    # never merged with the kinematic -2 (CLAUDE.md).
    v = _zlk(*_kinematic_vectors(), with_free=False).verdict
    assert v.status == Status.UNCERTAIN
    assert str(v.reason) == 'gate_unavailable'
    assert v.status_code == -9


def test_zlk_gate_disabled_keeps_the_geometric_libration():
    # A disabled gate is neither a confirmation nor a demotion; it is not one of the no-verdict outcomes
    # (decide.GATE_NO_VERDICT), so the geometry stands.
    off = SimpleNamespace(free_gate_enabled=False, free_gate_min_cycles=3.0)
    v = _zlk(*_kinematic_vectors(), gate_config=off).verdict
    assert v.status == Status.RESONANT
    assert v.status_code == 2


def test_zlk_circulating_omega_is_non_resonant_without_gate_change():
    # omega circulates (free vectors only, g = +50, s = -50 "/yr -> omega at 100"/yr, 386 turns): circulation (0)
    w = 50.0 * RAD_PER_ARCSEC * T_ZLK
    res = _zlk(0.2 * np.exp(1j * w), 0.1 * np.exp(-1j * w))
    assert res.verdict.status == Status.NON_RESONANT
    assert res.verdict.angle_status == AngleStatus.CIRCULATION
    assert res.verdict.status_code == 0


# --------------------------------------------------------------------------------------------------------------
# dry-run: pure formulas (SPEC §9.2)
# --------------------------------------------------------------------------------------------------------------


def test_orbital_and_pericentre_periods():
    assert dr.orbital_period_yr(4.0) == pytest.approx(8.0)
    assert dr.orbital_period_yr(1.0) == pytest.approx(1.0)
    # tau = P sqrt((1-e)^3/(1+e)): a = 4, e = 0.5 -> 8 sqrt(0.125/1.5) = 2.309401
    assert dr.tau_peri_yr(4.0, 0.5) == pytest.approx(2.309401, rel=1e-6)
    assert dr.tau_peri_yr(2.5, 0.0) == pytest.approx(2.5**1.5)


def test_hill_radii():
    # R_H = a (mu/3)^(1/3): Jupiter 5.2026 (9.546e-4/3)^(1/3) = 0.3552 au; Earth(+Moon) 1.0 (3.04e-6/3)^(1/3) = 0.01004
    assert dr.hill_radius_au('Jupiter') == pytest.approx(0.3552, rel=0.005)
    assert dr.hill_radius_au('Earth') == pytest.approx(0.01004, rel=0.01)


def test_crossing_planets_and_classes():
    # belt circle at 2.5 au crosses nothing
    assert dr.crossing_planets(2.5, 2.5, 1.0) == ()
    # q = 0.9, Q = 2.1: Earth [0.983, 1.017] and Mars [1.381, 1.666] are crossed; Venus (Q_V + R_H = 0.735) is not
    crossing = dr.crossing_planets(0.9, 2.1, 1.0)
    assert set(crossing) == {'Earth', 'Mars'}
    # Q = 4.5: Jupiter band q_J - h R_H = 4.951 - h x 0.355: 4.596 (h = 1) is not reached, 3.885 (h = 3) is
    assert 'Jupiter' not in dr.crossing_planets(4.0, 4.5, 1.0)
    assert 'Jupiter' in dr.crossing_planets(4.0, 4.5, 3.0)
    assert dr.body_class(0.9, ('Earth',), 1.3) == 'crosser'
    assert dr.body_class(1.1, (), 1.3) == 'nea'
    assert dr.body_class(2.0, (), 1.3) == 'main'


def test_recommended_integrator():
    dry = DryRunSettings()
    # a = 3, e = 0.5 (q = 1.5 >= 1.3, no crossing given): P = 3^1.5 = 5.196152,
    # tau = P sqrt(0.125/1.5) = sqrt(27/12) = 1.5 yr exactly; dt = min(1, P/10, tau/10) = 0.15 yr
    rec = dr.recommend_integrator(3.0, 0.5, (), 1.0, dry)
    assert rec['integrator'] == dr.SABA
    assert rec['dt_yr'] == pytest.approx(0.15, rel=1e-9)
    # q = 2.5 x 0.4 = 1.0 < DRYRUN_NEA_Q = 1.3: ias15 even without a crossing
    assert dr.recommend_integrator(2.5, 0.6, (), 1.0, dry)['integrator'] == dr.IAS15
    # e = 0: tau = P, dt = P/10 = 0.3952847
    assert dr.recommend_integrator(2.5, 0.0, (), 1.0, dry)['dt_yr'] == pytest.approx(0.3952847, rel=1e-6)
    # the run's own dt binds when smaller
    assert dr.recommend_integrator(2.5, 0.1, (), 0.05, dry)['dt_yr'] == pytest.approx(0.05)
    # a crosser goes to ias15
    assert dr.recommend_integrator(1.5, 0.4, ('Earth', 'Mars'), 1.0, dry)['integrator'] == dr.IAS15


def _laplace_hyp(s, j, alpha):
    # b_s^(j)(alpha) = 2 (s)_j / j! alpha^j 2F1(s, s + j; j + 1; alpha^2) (Murray & Dermott 1999, eq. 6.68)
    return 2 * poch(s, j) / factorial(j) * alpha**j * hyp2f1(s, s + j, j + 1, alpha * alpha)


@pytest.mark.parametrize('s,j,alpha', [(0.5, 0, 0.5), (1.5, 1, 0.5), (1.5, 1, 0.3), (1.5, 2, 0.8)])
def test_laplace_coefficient(s, j, alpha):
    assert dr.laplace_coefficient(s, j, alpha) == pytest.approx(_laplace_hyp(s, j, alpha), rel=1e-6)


def test_laplace_coefficient_elliptic_closed_form():
    # b_{1/2}^{(0)}(alpha) = (4/pi) K(alpha^2 as parameter m) = 2.146364 at alpha = 0.5
    assert dr.laplace_coefficient(0.5, 0, 0.5) == pytest.approx(4 / np.pi * ellipk(0.25), rel=1e-8)


def test_laplace_lagrange_free_precession_of_belt_particle():
    # A = (n/4) sum mu_j alpha_j alphabar_j b_{3/2}^{(1)}(alpha_j), computed here with IAU mass ratios and Standish
    # J2000 axes and the hypergeometric b: 44.50"/yr at 2.5 au, 75.71"/yr at 3.0 au.
    to_arcsec = 180 / np.pi * 3600
    assert dr.laplace_lagrange_precession(2.5) * to_arcsec == pytest.approx(44.50, rel=0.01)
    assert dr.laplace_lagrange_precession(3.0) * to_arcsec == pytest.approx(75.71, rel=0.01)


def test_snap_step():
    allowed = DryRunSettings().dryrun_auto_steps_yr
    assert dr.snap_step(3.75, allowed) == 2.5
    assert dr.snap_step(5.0, allowed) == 5.0  # <= : an allowed value is kept
    assert dr.snap_step(0.94, allowed) == 0.5
    assert dr.snap_step(0.03, allowed) == pytest.approx(0.03)  # below every allowed value: the limit itself
    assert dr.snap_step(float('inf'), allowed) == 500.0


def _orbit(a, e=0.0, inc=0.0, Omega=0.0, omega=0.0):
    return {'a': a, 'e': e, 'inc': inc, 'Omega': Omega, 'omega': omega}


def test_moid_closed_forms():
    # coplanar circles of 1 and 1.5 au: 0.5 au
    assert dr.moid_au(_orbit(1.5), _orbit(1.0)) == pytest.approx(0.5, abs=1e-6)
    # ellipse a = 2, e = 0.5 (q = 1) touching the unit circle at its perihelion: 0
    assert dr.moid_au(_orbit(2.0, 0.5), _orbit(1.0)) == pytest.approx(0.0, abs=1e-3)
    # a polar unit circle against the ecliptic unit circle: they cross at the nodes, 0
    assert dr.moid_au(_orbit(1.0, inc=np.pi / 2), _orbit(1.0)) == pytest.approx(0.0, abs=1e-3)
    # a polar circle of 3 au against a unit circle: the nearest points are on the line of nodes, 2 au
    assert dr.moid_au(_orbit(3.0, inc=np.pi / 2), _orbit(1.0)) == pytest.approx(2.0, abs=1e-3)


# --------------------------------------------------------------------------------------------------------------
# dry-run on a Simulation, static level (SPEC §9.1-§9.4)
# --------------------------------------------------------------------------------------------------------------

N_E, N_M = 360.000505, 191.406585  # deg per Gaussian year, the package model's mean motions (const.py)


def _n_deg(a):
    return 360.0 / a**1.5


def _sim(tmp_path, years=10000, nout=10001, **kwargs):
    return resonances.Simulation(
        integration_years=years,
        Nout=nout,
        save=None,
        save_summary=False,
        plot=None,
        save_path=str(tmp_path),
        plot_path=str(tmp_path),
        **kwargs,
    )


def _uq_like(sim, a0=1.0762):
    # 7E+2M-9 (1999 UQ): p = 0 by d'Alembert, no planet varpi: slow = 0, nu(a) = 7 n_E + 2 n_M - 9 n(a)
    sim.add_body({'a': a0, 'e': 0.0, 'inc': 0.01, 'Omega': 0.1, 'omega': 0.1, 'M': 0.1}, '7E+2M-9', name='uq')


def _belt(sim):
    sim.add_body({'a': 2.5, 'e': 0.1, 'inc': 0.05, 'Omega': 0.1, 'omega': 0.1, 'M': 0.1}, '3J-1', name='belt')


def test_static_row_of_a_three_body_resonance(tmp_path):
    sim = _sim(tmp_path)
    _uq_like(sim)
    rep = sim.dry_run('static')
    row = rep.rows.iloc[0]
    dt = float(row['step_yr'])
    assert dt == pytest.approx(1.0, rel=1e-3)
    # a*: n* = (7 n_E + 2 n_M)/9 = 322.535189 deg/yr -> a* = (360/322.535189)^(2/3) = 1.076012 au
    a_star = (360.0 / ((7 * N_E + 2 * N_M) / 9)) ** (2 / 3)
    assert row['a_res'] == pytest.approx(a_star, abs=2e-6)
    # rate at a0 and over a0 +- 0.001 au (NEA class: q = 1.0762 < 1.3)
    nu = lambda a: 7 * N_E + 2 * N_M - 9 * _n_deg(a)  # noqa: E731
    assert row['rate_deg_yr'] == pytest.approx(nu(1.0762), rel=1e-4)
    assert row['rate_min'] == pytest.approx(nu(1.0752), rel=1e-4)
    assert row['rate_max'] == pytest.approx(nu(1.0772), rel=1e-4)
    # Nyquist half-width a^(5/2) / (3 |k| dt) = 1.0762^2.5 / 27 = 0.044501 au
    assert row['nyquist_halfwidth_au'] == pytest.approx(1.0762**2.5 / (27 * dt), rel=1e-4)
    # ghosts n(a_m) = n* + 360 m / (k dt) with k = -9: m = -1 -> n = n* + 40, a = 0.995332 au (nearer than m = +1)
    a_m = (360.0 / ((7 * N_E + 2 * N_M) / 9 + 40.0 / dt)) ** (2 / 3)
    assert row['nearest_ghost_au'] == pytest.approx(abs(1.0762 - a_m), rel=1e-3)
    # safe steps: plain eta pi / max|nu| = 90 deg / 4.8026 deg/yr; predictive eta a^2.5 / (3 |k| da_sp)
    assert row['max_safe_step_plain'] == pytest.approx(90.0 / abs(nu(1.0772)), rel=1e-3)
    assert row['max_safe_step_pred'] == pytest.approx(0.5 * 1.0762**2.5 / (27 * 0.0003), rel=1e-3)
    assert row['verdict'] == 'resolved'  # |nu| dt = 4.8 deg << 90 deg
    body = rep.bodies.iloc[0]
    assert body['class'] == 'nea'
    assert body['integrator_recommended'] == dr.IAS15


def test_alias_branch_and_apparent_rate(tmp_path):
    # a0 = 1.08: nu = 7 n_E + 2 n_M - 9 x 360/1.08^1.5 = 2902.8167 - 2886.4... deg/yr; with a 20-yr step the branch
    # is m = round(nu dt / 360) and the apparent rate nu - 360 m / dt lies within +-180/dt = +-9 deg/yr.
    sim = _sim(tmp_path, years=20000, nout=1001)
    _uq_like(sim, a0=1.08)
    row = sim.dry_run('static').rows.iloc[0]
    dt = float(row['step_yr'])
    assert dt == pytest.approx(20.0, rel=1e-3)
    nu = 7 * N_E + 2 * N_M - 9 * _n_deg(1.08)
    m = round(nu * dt / 360.0)
    assert m != 0
    assert row['rate_deg_yr'] == pytest.approx(nu, rel=1e-4)
    assert row['apparent_deg_yr'] == pytest.approx(nu - 360.0 * m / dt, abs=1e-3)
    assert row['verdict'] != 'resolved'


def test_static_body_row_of_a_belt_asteroid(tmp_path):
    sim = _sim(tmp_path)
    _belt(sim)
    rep = sim.dry_run('static')
    body = rep.bodies.iloc[0]
    p = 2.5**1.5  # 3.952847 yr
    assert body['P_yr'] == pytest.approx(p, rel=1e-6)
    assert body['q'] == pytest.approx(2.25) and body['Q'] == pytest.approx(2.75)
    assert body['tau_peri_yr'] == pytest.approx(p * np.sqrt(0.9**3 / 1.1), rel=1e-6)
    assert body['fair_step_yr'] == pytest.approx(0.238 * p, rel=1e-6)
    assert body['class'] == 'main'
    assert body['integrator_recommended'] == dr.SABA
    assert body['varpi_rate_arcsec_yr'] == pytest.approx(44.50, rel=0.01)
    row = rep.rows.iloc[0]
    # 3J-1 -> 3J-1+0-2: k = -1, p = -2; jitter |p| de_sp / e with de_sp = da_sp / a = 0.003/2.5 = 0.0012: 0.024
    assert int(row['k']) == -1 and int(row['p']) == -2
    assert row['jitter'] == pytest.approx(2 * 0.0012 / 0.1, rel=1e-6)
    assert not bool(row['jitter_flag'])
    # baseline: (3/2) x 150 yr + 2 x 1 x P_c, P_c = 150 / 2 = 75 yr here -> 225 + 150 = 375 yr
    assert row['cutoff_pred_yr'] == pytest.approx(75.0)
    assert row['baseline_min_yr'] == pytest.approx(375.0, rel=1e-6)


def test_the_floor_cutoff_binds_at_any_step(tmp_path):
    # synodic period with Jupiter 1/(1/3.952847 - 30.349646/360) = 5.9284 yr, x 1.5 = 8.893 yr, below the floor bound
    # 150 / 2 = 75 yr: P_c = 75 yr at dt = 0.1 yr and at dt = 10 yr alike
    p_syn = 1.0 / (1.0 / 2.5**1.5 - 30.349646 / 360.0)
    assert 1.5 * p_syn == pytest.approx(8.893, abs=1e-3)
    for nout in (10001, 101):
        sim = _sim(tmp_path, years=1000, nout=nout)
        _belt(sim)
        row = sim.dry_run('static').rows.iloc[0]
        assert row['cutoff_pred_yr'] == pytest.approx(75.0) and row['cutoff_binding'] == 'floor'
        assert row['step_valid_yr'] == pytest.approx(37.5)


def test_auto_step_stays_below_the_short_period_nyquist_step(tmp_path):
    # Short periods: dt < min(P_orb = 3.952847 yr, P_syn,J = 5.9284 yr) / 2 = 1.976 yr -> snapped to 1 yr; the shortest
    # libration asks dt <= 150 / 20 = 7.5 yr (SPEC §9.4: P_lib,min / 20); the predictive-unwrap limit
    # 0.5 x 2.5^2.5 / (3 x 1 x 0.003) = 549 yr does not bind; FAIR is optional. Nout = 10000 / 1 + 1 = 10001.
    sim = _sim(tmp_path, output_step='auto')
    _belt(sim)
    rep = sim.dry_run('static')
    assert rep.run['step_limits_yr']['short_period'] == pytest.approx(2.5**1.5 / 2, rel=1e-6)
    assert rep.run['step_limits_yr']['libration'] == pytest.approx(7.5)
    assert rep.run['step_auto_yr'] == 1.0 and rep.run['binding_rule'] == 'short_period'
    assert rep.run['Nout_auto'] == round(rep.run['span_yr'] / 1.0) + 1
    rep.apply()
    assert sim.config.Nout == rep.run['Nout_auto']


def test_auto_step_with_fair_required(tmp_path):
    # FAIR: dt <= 0.238 P = 0.238 x 3.952847 = 0.9408 yr -> snapped 0.5 yr (SPEC §9.5); Nout = 10000/0.5 + 1
    sim = _sim(tmp_path, output_step='auto', dryrun_fair='required')
    _belt(sim)
    rep = sim.dry_run('static')
    assert rep.run['step_auto_yr'] == 0.5
    assert rep.run['Nout_auto'] == round(rep.run['span_yr'] / 0.5) + 1


def test_auto_step_user_cap(tmp_path):
    sim = _sim(tmp_path, output_step='auto', dryrun_max_step_yr=1.0)
    _belt(sim)
    assert sim.dry_run('static').run['step_auto_yr'] == 1.0


def test_secular_only_run_keeps_nout(tmp_path):
    # SPEC §9.4: 'auto' is defined for runs with MMRs only; a purely ZLK run keeps its Nout
    sim = _sim(tmp_path, years=100000, nout=5001, output_step='auto')
    sim.add_body({'a': 2.5, 'e': 0.2, 'inc': 0.5, 'Omega': 0.3, 'omega': 1.0, 'M': 0.5}, 'LK', name='zlk')
    rep = sim.dry_run('static')
    rep.apply()
    assert sim.config.Nout == 5001


def test_report_files(tmp_path):
    sim = _sim(tmp_path)
    _belt(sim)
    _uq_like(sim)
    rep = sim.dry_run('static')
    assert len(rep.rows) == 2 and len(rep.bodies) == 2
    paths = rep.to_csv(str(tmp_path / 'dry'))
    names = sorted(str(p).rsplit('/', 1)[-1] for p in paths.values())
    assert names == ['dryrun_bodies.csv', 'dryrun_rows.csv', 'dryrun_run.json']
    import pandas as pd

    assert len(pd.read_csv(tmp_path / 'dry' / 'dryrun_rows.csv')) == 2
    assert isinstance(rep.summary(echo=False), str)


def test_dry_run_changes_no_config_without_apply(tmp_path):
    sim = _sim(tmp_path, output_step='auto')
    _belt(sim)
    before = (sim.config.Nout, sim.config.dt)
    sim.dry_run('static')
    assert (sim.config.Nout, sim.config.dt) == before
