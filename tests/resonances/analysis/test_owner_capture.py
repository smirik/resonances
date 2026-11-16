"""A2 (W28, cards g05-6, g05-2): the second exclusion route F3 of the second capture, the secular-forcing fit it shares
with B10a, F2's well selection (a column only) and the slow motion of x with its cause. A2b (card g05-8 B): the third
outcome 'confirmed' (both resonances at once, origin joint) and 'one_cycle'.

The expected numbers come from outside the package: periods written by hand from the rates (pi / |omega_dot|,
2 pi / |varpi_dot - g_j|), a pendulum integrated with scipy and its period from the complete elliptic integral, and
Rayleigh lengths of centres placed by hand."""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.special import ellipk

from resonances.data import const
from resonances.resonance.analysis import filtering, rules
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import decide, kinematics, owner
from resonances.resonance.analysis.mmr.models import OwnerCandidate, OwnerResult, ResonanceSpec, SecondCapture, WellSelection
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import (
    AngleStatus,
    AngleSubtype,
    CauseRole,
    CheckValue,
    LibrationOrigin,
    OriginDetail,
    Reason,
    Status,
)

S = settings_from_config()
TWO_PI = 2 * math.pi
NAN = float('nan')
ARCSEC = math.pi / (180 * 3600)

# A 100-kyr record at 10 yr; 2 omega turns once per 11.6 kyr (omega_dot = pi / 11600 rad/yr), varpi at 40"/yr.
T = np.arange(0.0, 100000.0, 10.0)
TWO_OMEGA_PERIOD = 11600.0
OMEGA = math.pi / TWO_OMEGA_PERIOD * T
VARPI = 40.0 * ARCSEC * T
P_OWNER = 200.0  # the owner-band period: Psi is low-passed at 3 P = 600 yr
SPAN = (0.0, float(T[-1]))


def _secular():
    return OMEGA, kinematics.secular_varpi_terms(T, VARPI)


def _capture(x, uv=(1, 1), phi=None, secular=True):
    """`second_capture` of a Psi whose x = Psi / u is `x`: sigma = (Psi + v phi) / u."""
    u, v = uv
    phi = np.zeros_like(T) if phi is None else phi
    sigma = (u * np.asarray(x) + v * phi) / u
    return owner.second_capture(T, sigma, phi, uv, SPAN, P_OWNER, 1.0, S, _secular() if secular else None)


# --------------------------------------------------------------------------------------------- secular_forcing_fit
def test_secular_forcing_fit_names_the_term_and_its_period():
    # by hand: pi / omega_dot = 11600 yr; 2 pi / |40" - g5| = 1296000 / (40 - 4.25749319) = 36259.3 yr
    y = 1.0 + 3.0 * np.cos(2 * OMEGA - 0.4) - 0.5 * np.sin(4 * OMEGA)
    r2, period, term = kinematics.secular_forcing_fit(T, y, OMEGA, kinematics.secular_varpi_terms(T, VARPI), 2)
    assert term == '2omega' and r2 == pytest.approx(1.0, abs=1e-9) and period == pytest.approx(11600.0, rel=1e-9)
    y = 2.0 * np.sin(VARPI - const.PLANETARY_FREQUENCIES['g5'] * ARCSEC * T + 1.1)  # any phase: the fit absorbs it
    r2, period, term = kinematics.secular_forcing_fit(T, y, OMEGA, kinematics.secular_varpi_terms(T, VARPI), 2)
    assert term == 'g-g5' and r2 == pytest.approx(1.0, abs=1e-9) and period == pytest.approx(36259.3, rel=1e-5)
    assert kinematics.secular_cause(term) == 'secular_g_gj' and kinematics.secular_cause('2omega') == 'secular_2omega'


def test_secular_forcing_terms_need_samples_and_a_varying_y():
    terms = kinematics.secular_forcing_terms(T[:10], np.ones(10), OMEGA[:10], {}, 2, samples_per_term=3)
    assert set(terms) == {'2omega'} and math.isnan(terms['2omega'][0])
    assert kinematics.secular_forcing_fit(T, np.ones_like(T), OMEGA, {}, 2) == (NAN, NAN, '') or math.isnan(
        kinematics.secular_forcing_fit(T, np.ones_like(T), OMEGA, {}, 2)[0]
    )


def test_a_term_longer_than_the_span_allows_is_not_tested():
    # a slow parabola over 20 kyr: cos and sin of a 150-kyr phase fit it almost exactly (a low polynomial in t), which
    # says nothing about a secular cause; with a period limit of 12 kyr that term is not tested
    t = T[T < 20000.0]
    y = (t / 1e4) ** 2
    theta = {'g-g8': TWO_PI * t / 150000.0}
    loose = kinematics.secular_forcing_terms(t, y, OMEGA[: len(t)], theta, 2)
    assert loose['g-g8'][0] > 0.99
    strict = kinematics.secular_forcing_terms(t, y, OMEGA[: len(t)], theta, 2, period_max_yr=12000.0)
    assert math.isnan(strict['g-g8'][0]) and strict['g-g8'][1] == pytest.approx(150000.0)
    assert strict['2omega'][1] == pytest.approx(TWO_OMEGA_PERIOD) and np.isfinite(strict['2omega'][0])  # 11.6 kyr fits


# --------------------------------------------------------------------------------------------- route F3
def _forced_x(drift_deg=100.0, wave_deg=20.0):
    """x = a monotonic drift + a 2 omega wave: x_ddot = -(2 omega_dot)^2 A cos 2 omega, nothing of sin x."""
    return math.radians(drift_deg) * T / T[-1] + math.radians(wave_deg) * np.cos(2 * OMEGA)


def test_x_driven_by_the_double_omega_cycle_is_excluded_by_f3_through_owner_slips():
    # 78085-like: u = 7, v = 1; phi_o slips by whole turns (three 2 pi steps of 300 yr): Psi = 7 sigma - phi has no slip
    phi = sum(TWO_PI * 0.5 * (1 + np.tanh((T - t0) / 100.0)) for t0 in (20000.0, 45000.0, 70000.0))
    capture = _capture(_forced_x(), uv=(7, 1), phi=phi)
    assert math.degrees(capture.x_sweep_rad) < S.mmr_owner_capture_sweep_deg  # the sweep route cannot exclude
    assert capture.secular_r2 >= 0.99 and capture.self_r2 < S.mmr_owner_capture_self_r2_max
    assert capture.verdict == owner.CAPTURE_EXCLUDED and capture.route == owner.ROUTE_SECULAR
    columns = capture.to_columns()
    assert columns['own_second_capture'] == 'excluded' and columns['own_second_capture_route'] == 'secular_forcing'
    # without the secular phases F3 is not measured and the same x stays unresolved
    blind = _capture(_forced_x(), uv=(7, 1), phi=phi, secular=False)
    assert math.isnan(blind.secular_r2) and blind.verdict == owner.CAPTURE_UNRESOLVED and blind.route == ''


def _pendulum(amplitude_rad, own_period_yr):
    w = TWO_PI / own_period_yr

    def rhs(_, y):
        return [y[1], -(w**2) * math.sin(y[0])]

    return solve_ivp(rhs, (0, T[-1]), [amplitude_rad, 0.0], t_eval=T, rtol=1e-10, atol=1e-12, max_step=20.0).y[0]


def test_a_free_pendulum_in_x_is_not_excluded_and_its_period_is_read():
    amplitude, small_period = math.radians(80.0), 4500.0
    capture = _capture(_pendulum(amplitude, small_period))
    # not excluded by either route; since A2b a free pendulum in x is the confirmed joint capture (tests below)
    assert capture.self_r2 > 0.9 and capture.verdict == owner.CAPTURE_CONFIRMED and capture.route == ''
    # the period of a pendulum of amplitude A: T = (2 / pi) K(sin^2(A / 2)) T_small (scipy's ellipk takes m = k^2)
    expected = 2.0 / math.pi * ellipk(math.sin(amplitude / 2) ** 2) * small_period  # 5097 yr
    motion = capture.slow_x
    assert motion.period_yr == pytest.approx(expected, rel=0.03)
    assert math.degrees(motion.halfamp_rad) == pytest.approx(80.0, rel=0.1)  # its fundamental carries ~ A (sin^3 terms aside)
    assert motion.cause == '' and motion.level == ''  # no secular period near 5.1 kyr, no secular x_dot


def test_a_monotonic_sweep_of_150_deg_stays_unresolved():
    rng = np.random.default_rng(11)
    x = math.radians(150.0) * T / T[-1] + math.radians(0.5) * rng.normal(size=len(T))
    capture = _capture(x)
    assert 140.0 < math.degrees(capture.x_sweep_rad) < S.mmr_owner_capture_sweep_deg
    assert not capture.secular_r2 >= S.mmr_owner_r2_min
    assert capture.verdict == owner.CAPTURE_UNRESOLVED and capture.route == ''
    # a noiseless ramp: x_ddot is round-off, no R^2 is read from it
    ramp = _capture(math.radians(150.0) * T / T[-1])
    assert math.isnan(ramp.self_r2) and math.isnan(ramp.secular_r2) and ramp.verdict == owner.CAPTURE_UNRESOLVED


def test_the_sweep_route_still_excludes_and_takes_the_name():
    # x turns about 3 times at a rate modulated by a random walk: no energy integral (the W20 test of the first route)
    rng = np.random.default_rng(5)
    walk = filtering.lowpass(T, np.cumsum(rng.normal(size=len(T))), 3000.0, 2)
    walk = (walk - walk.mean()) / np.max(np.abs(walk - walk.mean()))
    rate = math.radians(1100.0) / T[-1] * (1 + 0.5 * walk)
    x = np.concatenate([[0.0], np.cumsum(rate[1:] * np.diff(T))])
    capture = _capture(x)
    assert capture.verdict == owner.CAPTURE_EXCLUDED and capture.route == owner.ROUTE_SWEEP_ENERGY


# --------------------------------------------------------------------------------------------- slow x and its cause
def test_slow_x_cause_identified_possible_and_none():
    identified = _capture(_forced_x(drift_deg=30.0)).slow_x
    assert identified.period_yr == pytest.approx(TWO_OMEGA_PERIOD, rel=S.mmr_kin_period_tol)
    assert identified.secular_period_yr == pytest.approx(TWO_OMEGA_PERIOD, rel=1e-6)
    assert identified.term == '2omega' and identified.cause == 'secular_2omega' and identified.level == 'identified'
    assert math.degrees(identified.halfamp_rad) == pytest.approx(20.0, rel=0.05)
    # the 2 omega wave still dominates x (20 deg against 10 deg), but x_dot is mostly a 3-kyr wave:
    # by hand the 2 omega share of var(x_dot) is (20 / 11600)^2 / ((20 / 11600)^2 + (10 / 3000)^2) = 0.21 < 0.7
    x = math.radians(20.0) * np.cos(2 * OMEGA) + math.radians(10.0) * np.sin(TWO_PI * T / 3000.0)
    possible = _capture(x).slow_x
    assert possible.period_yr == pytest.approx(TWO_OMEGA_PERIOD, rel=S.mmr_kin_period_tol)
    assert possible.secular_r2 == pytest.approx(0.21, abs=0.05)
    assert possible.cause == 'secular_2omega' and possible.level == 'possible'
    none = _capture(_pendulum(math.radians(60.0), 4500.0)).slow_x
    assert none.cause == '' and none.level == '' and none.to_columns()['own_x_cause_level'] == ''


def test_the_columns_of_an_untested_capture_are_empty():
    empty = SecondCapture.empty_columns()
    for column in ('own_psi_self_R2', 'own_psi_secular_R2', 'own_x_period_yr', 'own_x_halfamp_deg', 'own_x_secular_R2'):
        assert empty[column] is None
    for column in ('own_second_capture_route', 'own_x_cause', 'own_x_cause_level', 'own_x_secular_term'):
        assert empty[column] == ''
    assert WellSelection.empty_columns() == {
        'own_capture_R_owner': None,
        'own_capture_R_sigma': None,
        'own_capture_n_recaptures': None,
    }


# --------------------------------------------------------------------------------------------- F2 well selection
def test_well_selection_reads_the_rayleigh_lengths_of_the_centres():
    # four episodes of sigma, its centres at 0, 90, 180, 270 deg (R = 0 by hand); phi_o back at 180 deg each time (R = 1)
    centres = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)
    sigma, phi = np.zeros_like(T), np.full_like(T, math.pi)
    windows = []
    for k, centre in enumerate(centres):
        t0, t1 = 5000.0 + 25000.0 * k, 20000.0 + 25000.0 * k
        inside = (T >= t0 - 2500.0) & (T < t1 + 2500.0)
        sigma[inside] = centre
        windows.append((k, t0, t1, 1000.0))
    wells = owner.well_selection(T, sigma, phi, windows, [SPAN], P_OWNER, S)
    assert wells.r_sigma == pytest.approx(0.0, abs=1e-3) and wells.r_owner == pytest.approx(1.0, abs=1e-6)
    assert wells.n_recaptures == 3
    # only the episodes that meet the test span count; one episode leaves R unmeasured
    one = owner.well_selection(T, sigma, phi, windows, [(0.0, 21000.0)], P_OWNER, S)
    assert one.n_recaptures == 0 and math.isnan(one.r_owner) and math.isnan(one.r_sigma)


# --------------------------------------------------------------------------------------------- the decision
def _owner_result(capture):
    accepted = OwnerCandidate(
        spec=ResonanceSpec.from_key('2E-7+0+5'),
        rate_rad_yr=0.0,
        r2_median=0.9,
        r2_windows=(0.9,) * 10,
        phi0_coherence=0.97,
        psi_share=0.0,
        force_au_yr=1.0,
        force_theory_au_yr=1.0,
        strength_ratio=0.01,
        rank=1,
        accepted=True,
        reason='accepted:r2',
        route='r2',
    )
    return OwnerResult(
        1, [accepted], accepted, CheckValue.FAIL, 0.0, NAN, NAN, NAN, NAN, NAN, 1718.0, 'owner_libration', cover=1.0, capture=capture
    )


def _capture_numbers(route, verdict):
    # 78085 5J-4S-1: x sweeps 155 deg, energy R^2 0.05, secular R^2 0.98, self R^2 0.001
    return SecondCapture(1.2, math.radians(155.1), 0.05, 0.02, verdict, self_r2=0.001, secular_r2=0.98, route=route)


def test_f3_removes_the_cap_and_fires_its_rule():
    angle = decide.AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, 'G2', 0)
    result = _owner_result(_capture_numbers(owner.ROUTE_SECULAR, owner.CAPTURE_EXCLUDED))
    ledger = Ledger.from_settings(S)
    decision = decide._foreign(angle, None, result, S, ledger)
    assert decision.status == Status.NON_RESONANT and decision.reason == Reason.INDUCED_LIBRATION
    assert decision.caps == [] and 'second_capture_unresolved' not in decision.flags
    assert rules.Rule.SECOND_CAPTURE_F3 in decision.rules_fired
    entries = {entry.name: entry for entry in ledger}
    assert entries['2 second capture: secular R2'].setting == 'mmr_owner_r2_min'
    assert entries['2 second capture: self R2'].setting == 'mmr_owner_capture_self_r2_max'
    assert entries['2 second capture: self R2'].result is True


def test_an_unresolved_capture_keeps_the_cap_and_its_f3_numbers_decide_nothing():
    angle = decide.AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, 'G2', 0)
    numbers = SecondCapture(1.2, math.radians(155.1), 0.05, 0.02, owner.CAPTURE_UNRESOLVED, self_r2=0.19, secular_r2=0.69)
    ledger = Ledger.from_settings(S)
    decision = decide._foreign(angle, None, _owner_result(numbers), S, ledger)
    assert decision.caps == ['second_capture_unresolved'] and rules.Rule.SECOND_CAPTURE_F3 not in decision.rules_fired
    f3 = [entry for entry in ledger if entry.name in ('2 second capture: secular R2', '2 second capture: self R2')]
    assert len(f3) == 2 and all(entry.setting == '' for entry in f3)  # recorded, never a thin margin
    assert not any('secular R2' in name or 'self R2' in name for name in ledger.thin_margins())
    # the sweep route excluding: no F3 rule
    swept = _owner_result(SecondCapture(3.0, math.radians(243.0), 0.26, 0.03, owner.CAPTURE_EXCLUDED, route=owner.ROUTE_SWEEP_ENERGY))
    decision = decide._foreign(angle, None, swept, S, Ledger.from_settings(S))
    assert decision.caps == [] and rules.Rule.SECOND_CAPTURE_F3 not in decision.rules_fired


# --------------------------------------------------------------------------------------------- A2b: both resonances at once
# The secular periods of this record by hand (varpi_dot = 40"/yr, k = 1, 2): pi / omega_dot = 11600 and 5800 yr;
# 2 pi / |40" - g_j| = 1296000 / (40 - g_j) yr: g5 36259 / 18130, g6 110224 / 55112, g7 35111 / 17556, g8 32953 / 16476.
SECULAR_BY_HAND = (11600.0, 5800.0, 36259.3, 18129.7, 110224.0, 55112.0, 35111.0, 17555.5, 32953.0, 16476.5)


def _offset_by_hand(period_yr):
    return min(abs(period_yr / reference - 1) for reference in SECULAR_BY_HAND)


def _two_pendula(amplitude_deg=80.0, small_period=4500.0, uv=(1, 2)):
    """Two independent pendula sharing lambda: phi_o librates on its own (60-yr period, 25 deg), sigma has its own fast
    libration (75 yr, 30 deg) and x = Psi / u = sigma - (v / u) phi_o swings in a well of its own (a pendulum)."""
    u, v = uv
    phi = 0.4 + math.radians(25.0) * np.sin(TWO_PI * T / 60.0)
    x = _pendulum(math.radians(amplitude_deg), small_period)
    sigma = v / u * phi + x + math.radians(30.0) * np.sin(TWO_PI * T / 75.0)
    return owner.second_capture(T, sigma, phi, uv, SPAN, P_OWNER, 1.0, S, _secular())


def test_two_independent_pendula_are_confirmed():
    capture = _two_pendula()
    amplitude = math.radians(80.0)
    expected = 2.0 / math.pi * ellipk(math.sin(amplitude / 2) ** 2) * 4500.0  # the pendulum's period, 5097 yr
    assert capture.verdict == owner.CAPTURE_CONFIRMED and capture.route == ''
    # the fast librations leak a little through the 3 P low-pass into x_ddot (self R^2 0.94; 1.0 without them)
    assert capture.energy_r2 >= 0.99 and capture.self_r2 >= S.mmr_owner_r2_min
    assert math.degrees(capture.x_sweep_rad) == pytest.approx(160.0, abs=2.0)  # 2 x 80 deg, below 180
    assert capture.x_cycle_period_yr == pytest.approx(expected, rel=0.01)
    # 100 kyr / 5.1 kyr = 19.6 cycles; the first and last half-cycles touch the record's edges
    assert 17 <= capture.x_cycles <= 19
    # the nearest secular period is the second harmonic of 2 omega, 5800 yr: offset 0.12 > mmr_kin_period_tol
    assert capture.x_secular_offset == pytest.approx(_offset_by_hand(capture.x_cycle_period_yr), abs=1e-3)
    assert capture.x_secular_offset == pytest.approx(1 - expected / 5800.0, abs=0.01)
    columns = capture.to_columns()
    assert columns['own_second_capture'] == 'confirmed' and columns['own_x_cycles'] == capture.x_cycles


def test_x_forced_at_two_omega_with_the_same_spectrum_is_never_confirmed():
    # the record's own 2 omega (11.6 kyr) forces x with an 80-deg wave and its second harmonic: x is bounded and closes
    # cycles, and a forced sinusoid passes the energy fit like a pendulum; its period is the secular one
    for x in (
        math.radians(80.0) * np.cos(2 * OMEGA),
        math.radians(60.0) * np.cos(2 * OMEGA) + math.radians(25.0) * np.cos(4 * OMEGA + 0.7),
    ):
        capture = _capture(x)
        assert capture.x_cycles >= S.libration_centre_min_estimates
        assert capture.x_secular_offset <= S.mmr_kin_period_tol
        assert capture.verdict != owner.CAPTURE_CONFIRMED and capture.verdict != owner.CAPTURE_ONE_CYCLE


def test_a_monotonic_x_of_170_deg_is_never_confirmed():
    # a single-valued x_dot(x) fits the energy integral of any monotonic x (2013 TW135: R^2 1.00): no turn, no cycle
    rng = np.random.default_rng(3)
    for x in (math.radians(170.0) * T / T[-1], math.radians(170.0) * (T / T[-1]) ** 2 + math.radians(0.3) * rng.normal(size=len(T))):
        capture = _capture(x)
        assert math.degrees(capture.x_sweep_rad) < S.mmr_owner_capture_sweep_deg
        assert capture.x_cycles == 0 and capture.verdict == owner.CAPTURE_UNRESOLVED


def test_one_cycle_of_its_own_well_is_one_cycle():
    # small period 39 kyr at 80 deg: T = (2 / pi) K(sin^2 40 deg) x 39 kyr = 44.1 kyr; over 100 kyr from a turning point
    # the turns at 22, 44, 66, 88 kyr: one independent cycle; 44.1 kyr is no secular period (nearest 36.3 kyr, 0.22)
    expected = 2.0 / math.pi * ellipk(math.sin(math.radians(40.0)) ** 2) * 39000.0
    capture = _capture(_pendulum(math.radians(80.0), 39000.0))
    assert capture.x_cycles == 1 and capture.x_cycle_period_yr == pytest.approx(expected, rel=0.02)
    assert capture.x_secular_offset == pytest.approx(_offset_by_hand(capture.x_cycle_period_yr), abs=1e-3)
    assert capture.verdict == owner.CAPTURE_ONE_CYCLE


def test_joint_outcome_needs_every_condition():
    good = dict(sweep_rad=math.radians(120.0), energy_r2=0.95, self_r2=0.9, cycles=3, offset=0.3)
    assert owner.joint_outcome(settings=S, **good) == owner.CAPTURE_CONFIRMED
    assert owner.joint_outcome(settings=S, **{**good, 'cycles': 1}) == owner.CAPTURE_ONE_CYCLE
    for change in (
        {'sweep_rad': math.radians(181.0)},
        {'energy_r2': 0.69},
        {'self_r2': 0.69},
        {'offset': 0.1},
        {'offset': NAN},
        {'self_r2': NAN},
        {'cycles': 0},
    ):
        assert owner.joint_outcome(settings=S, **{**good, **change}) == ''


def test_blocks_confirm_only_together():
    t = T
    pendulum = _pendulum(math.radians(80.0), 4500.0)
    ramp = math.radians(150.0) * t / t[-1]
    x = np.where(t < 50000.0, pendulum, pendulum[t < 50000.0][-1] + ramp - ramp[t >= 50000.0][0])
    blocks = [(0.0, 49000.0), (51000.0, float(t[-1]))]
    capture = owner.second_capture_blocks(t, x, np.zeros_like(t), (1, 1), blocks, P_OWNER, 1.0, S, _secular())
    assert capture.verdict == owner.CAPTURE_UNRESOLVED  # the ramp block neither confirms nor excludes
    both = owner.second_capture_blocks(t, pendulum, np.zeros_like(t), (1, 1), blocks, P_OWNER, 1.0, S, _secular())
    assert both.verdict == owner.CAPTURE_CONFIRMED


def _joint_numbers(verdict, cycles=12):
    return SecondCapture(
        -0.4,
        math.radians(160.0),
        0.999,
        0.02,
        verdict,
        self_r2=0.999,
        secular_r2=0.05,
        x_cycles=cycles,
        x_cycle_period_yr=5097.0,
        x_secular_offset=0.12,
    )


def test_foreign_steps_aside_on_a_confirmed_capture_and_names_one_cycle_uncertain():
    angle = decide.AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, 'G2', 0)
    ledger = Ledger.from_settings(S)
    assert decide._foreign(angle, None, _owner_result(_joint_numbers(owner.CAPTURE_CONFIRMED)), S, ledger) is None
    entries = {entry.name: entry for entry in ledger}
    assert entries[decide.JOINT_CYCLES].setting == 'libration_centre_min_estimates' and entries[decide.JOINT_CYCLES].result
    assert entries[decide.JOINT_OFFSET].setting == 'mmr_kin_period_tol'
    one = decide._foreign(angle, None, _owner_result(_joint_numbers(owner.CAPTURE_ONE_CYCLE, cycles=1)), S, Ledger.from_settings(S))
    assert (one.status, one.reason, one.origin, one.origin_detail, one.owner, one.rule) == (
        Status.UNCERTAIN,
        Reason.BASELINE_NEEDED,
        LibrationOrigin.JOINT,
        OriginDetail.JOINT_CAPTURE,
        '2E-7+0+5',
        '2j',
    )
    assert rules.Rule.BOTH_RESONANCES_CONFIRMED in one.rules_fired and one.caps == []
    # an unresolved capture: the joint conditions are on the ledger without their settings (no thin margin)
    ledger = Ledger.from_settings(S)
    induced = decide._foreign(angle, None, _owner_result(_joint_numbers(owner.CAPTURE_UNRESOLVED, cycles=0)), S, ledger)
    assert induced.reason == Reason.INDUCED_LIBRATION and induced.origin == LibrationOrigin.INDUCED
    assert all(entry.setting == '' for entry in ledger if entry.name.startswith('2j'))


def _decide_episodic(capture):
    """decide_status of an episodic angle whose own force law and potential pass (the 5d/6 dynamic leaf) and whose
    accepted owner covers its episode (rule 2 without A2b)."""
    from tests.resonances.analysis.builders import make, make_episode, make_geometry, make_response, make_sampling
    from resonances.resonance.analysis.mmr.models import EpisodePhysics
    from resonances.resonance.analysis.pendulum import EnergyAnalysis

    result = _owner_result(capture)
    result.intervals = [(0.0, 9000.0, 'libration')]
    result.decisive_window = (1000.0, 8000.0)
    result.cover = owner.covered_fraction(result.intervals, result.decisive_window)
    owner.cover_episodes(result, [(0, 1000.0, 8000.0, 1000.0)], 1000.0)
    times = [1000.0 + 500.0 * k for k in range(15)]
    geometry = make_geometry(window=(0.0, 9000.0), runs=[make_episode(episode_id=0, tp_times_yr=times, period_med_yr=1000.0)])
    response = make_response(
        a_response=CheckValue.NOT_MEASURED,
        a_amplitude=CheckValue.PASS,
        slow_libration=CheckValue.NOT_APPLICABLE,
        identity=CheckValue.PASS,
        b1_ratio=0.9,
        force_r2_h1=NAN,
        force_r2_h2=NAN,
        force_contaminated=False,
    )
    sampling = make_sampling(
        unwrap_resid_p99_rad=0.0,
        unwrap_resid_max_rad=0.0,
        unwrap_ok=True,
        undefined_phase=False,
        verdict='resolved',
        alias_numpy_differs=False,
    )
    energy = EnergyAnalysis(None, None, 1.0, 1.0, with_slow=CheckValue.PASS)
    physics = [make(EpisodePhysics, episode_id=0, a_band_std_before_au=0.1, a_band_std_in_au=1.0)]
    angle = decide.AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, 'G2', 0)
    return decide.decide_status(
        angle, sampling, geometry, energy, response, None, None, result, None, S, Ledger.from_settings(S), episodes=physics
    )


def test_a_confirmed_capture_keeps_the_own_reading_with_origin_joint():
    induced = _decide_episodic(_joint_numbers(owner.CAPTURE_UNRESOLVED, cycles=0))
    assert (induced.status, induced.origin, induced.rule) == (Status.NON_RESONANT, LibrationOrigin.INDUCED, '2')
    joint = _decide_episodic(_joint_numbers(owner.CAPTURE_CONFIRMED))
    assert joint.status == Status.TRANSIENT and joint.reason == Reason.DYNAMIC_EPISODE  # the tree's own reading
    assert joint.origin == LibrationOrigin.JOINT and joint.origin_detail == OriginDetail.JOINT_CAPTURE
    assert joint.owner == '2E-7+0+5' and rules.Rule.BOTH_RESONANCES_CONFIRMED in joint.rules_fired
    assert 'second_capture_unresolved' not in joint.flags and decide.own_dynamics_claimed(joint)
    # the roles: the other resonance is named as the owner with the role joint
    from resonances.resonance.analysis.mmr import roles

    named = roles.verdict_roles(
        ResonanceSpec.from_key('5J-4S-1+0+0+0'), joint.origin, joint.origin_detail, joint.owner, joint.flags, None, S
    )
    assert named.owner == '2E-7+0+5' and named.cause_role == CauseRole.JOINT


def test_the_joint_origin_reaches_the_csv_and_the_save_filters():
    from resonances.simulation.config import status_selected

    assert LibrationOrigin('joint') == LibrationOrigin.JOINT and OriginDetail('joint_capture') == OriginDetail.JOINT_CAPTURE
    for mode in ('nonzero', 'extended', 'candidates'):
        assert status_selected(mode, 'transient', 'joint')
    assert status_selected('nonzero', 'uncertain', 'joint') and not status_selected('resonant', 'uncertain', 'joint')
    assert SecondCapture.empty_columns()['own_x_cycles'] is None


# --------------------------------------------------------------------------------------------- benchmark rows
def analyse_example(example):
    from resonances.resonance.analysis.mmr import io
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    if not (io.example_path(example) / 'series.csv.gz').exists():
        pytest.skip('benchmark data not available')
    return analyse_mmr(io.load_series(example), S)


@pytest.mark.benchmark
def test_78085_is_excluded_by_f3_and_hw7_stays_unresolved():
    a = analyse_example('mmr3__78085__5J-4S-1+0+0+0')
    numbers = a.numbers_row()
    assert numbers['own_second_capture'] == 'excluded' and numbers['own_second_capture_route'] == 'secular_forcing'
    assert numbers['own_psi_secular_R2'] >= S.mmr_owner_r2_min and numbers['own_psi_self_R2'] < S.mmr_owner_capture_self_r2_max
    assert str(a.verdict.status) == 'non_resonant' and str(a.verdict.confidence) == 'medium'
    assert 'second_capture_unresolved' not in a.verdict.flags and 'second_capture_f3' in a.verdict.rules_fired
    assert numbers['own_capture_R_owner'] > 0.9 and numbers['own_capture_R_sigma'] < 0.5  # F2: a column only

    hw7 = analyse_example('mmr3__2001_HW7__5J+6S-2+0+0-9')
    numbers = hw7.numbers_row()
    assert numbers['own_second_capture'] == 'unresolved' and 'second_capture_unresolved' in hw7.verdict.flags
    assert str(hw7.verdict.reason) == 'induced_libration' and hw7.verdict.owner == '5E-16+0+11'
    # card g05-2 (tier C): the slow x of HW7 turns at the double omega cycle, pi / |omega_dot|
    assert abs(numbers['own_x_period_yr'] / numbers['kin_two_omega_period_yr'] - 1) <= 0.25
    assert numbers['own_x_cause'] == 'secular_2omega' and numbers['own_x_cause_level'] in ('possible', 'identified')
    # A2b: neither row is confirmed (78085 x has no turn, HW7 one turn and energy R^2 0.25)
    assert numbers['own_second_capture'] not in ('confirmed', 'one_cycle')
    assert str(a.verdict.libration_origin) == 'induced' and str(hw7.verdict.libration_origin) == 'induced'
