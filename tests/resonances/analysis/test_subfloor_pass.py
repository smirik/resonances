"""Rule G1f (W28 B5, card g08-2 B): a two-body cycle below the period floor, confirmed by the resonance's own averaged
potential, its centre and its force law.

Independent sources: the small-amplitude period of 2001 VC2 16E-17 measured on its N-body record (T0 92.7-102.6 yr,
the episode at a 60-yr floor), and the finite-amplitude period of a pendulum, T(A) = T0 2 K(sin(A/2)) / pi (complete
elliptic integral, scipy), which the synthetic record must show."""

import dataclasses
import itertools
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.special import ellipk

from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import analyse as analyse_module
from resonances.resonance.analysis.mmr import context as context_module
from resonances.resonance.analysis.mmr import decide, models
from resonances.resonance.analysis.mmr import preprocess, sampling as sampling_module, strength as strength_module
from resonances.resonance.analysis.mmr.context import mean_motion
from resonances.resonance.analysis.mmr.models import MMRSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, CheckValue, ResonanceType, Status, Verdict
from tests.resonances.analysis.builders import make, make_episode, make_geometry, make_sampling

SETTINGS = settings_from_config()
VC2_KEY = '16E-17+0+1'  # Earth 16:17, first order; mean elements of 2001 VC2 below
A_VC2, E_VC2, INC_VC2, OMEGA_VC2, NODE_VC2 = 1.0412, 0.131, 0.2027, 1.73, -0.10


def pendulum_record(key=VC2_KEY, period_yr=100.0, amplitude_deg=60.0, centre_rad=0.0, span_yr=20000.0, step_yr=5.0):
    """sigma_ddot = -w^2 sin(sigma - centre), w = 2 pi / T0, a from the identity a = a* + sigma_dot / c (so a_dot follows
    the force law), the other elements fixed at 2001 VC2's means."""
    spec = ResonanceSpec.from_key(key)
    t = np.arange(0.0, span_yr + step_yr / 2, step_yr)
    w = 2 * np.pi / period_yr
    solution = solve_ivp(
        lambda _, y: [y[1], -w * w * np.sin(y[0])],
        (0.0, t[-1]),
        [math.radians(amplitude_deg), 0.0],
        t_eval=t,
        method='DOP853',
        rtol=1e-10,
        atol=1e-12,
    )
    angle, rate = solution.y
    c = -1.5 * spec.body_coefficient * mean_motion(A_VC2) / A_VC2
    n = len(t)
    omega, node = np.full(n, OMEGA_VC2), np.full(n, NODE_VC2)
    return MMRSeries(
        t,
        A_VC2 + rate / c,
        np.full(n, E_VC2),
        np.full(n, INC_VC2),
        node,
        omega,
        omega + node,
        np.mod(angle + centre_rad, 2 * np.pi),
        spec,
        Regime(step_yr, span_yr, n),
    )


def forced_record(key='5J-2S-2+0+0-1', period_yr=118.0, amplitude_deg=40.0, span_yr=20000.0, step_yr=5.0):
    """A forced wave of a three-body angle (52356, 23240: 115-121 yr): a sinusoid in sigma, a constant."""
    spec = ResonanceSpec.from_key(key)
    t = np.arange(0.0, span_yr + step_yr / 2, step_yr)
    n = len(t)
    angle = math.radians(amplitude_deg) * np.sin(2 * np.pi * t / period_yr)
    zero = np.zeros(n)
    return MMRSeries(
        t, np.full(n, 3.17), np.full(n, 0.1), zero, zero, zero, zero, np.mod(angle, 2 * np.pi), spec, Regime(step_yr, span_yr, n)
    )


def first_pass(series):
    context = context_module.build_context(series, SETTINGS)
    sampling = sampling_module.analyse_sampling(series, context, SETTINGS)
    return context, sampling, preprocess.preprocess(series, context, sampling, SETTINGS)


# --- eligibility -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize('key', ['5J-2S-2+0+0-1', '1M+3J-3+0+0-1', '1E-1+0+0', '1J-1+0+0'])
def test_three_body_and_one_to_one_have_no_profile_floor(key):
    """Three-body: no computed theory; 1:1: the co-orbital branch is C3's. The hook needs nothing else to say so."""
    series = forced_record(key)
    assert not strength_module.direct_applicable(series.resonance)
    assert preprocess.profile_period_floor(series, None, None, SETTINGS) is None


def test_vc2_theory_agrees_with_its_n_body_period():
    """T0_th from VC2's mean elements against the T0 its own N-body record shows (92.7-102.6 yr): F's band is a factor
    of 10 in strength, sqrt(10) in period; the theory sits within 1.5 of the record."""
    series = pendulum_record()
    context, _, prep = first_pass(series)
    found = preprocess.profile_period_floor(series, context, prep, SETTINGS)
    assert found is not None and found.source == preprocess.SUBFLOOR_SOURCE
    assert 92.7 / 1.5 < found.period_yr < 102.6 * 1.5
    assert found.floor_yr == pytest.approx(found.period_yr / math.sqrt(SETTINGS.mmr_strength_check_ratio))


def test_no_profile_floor_without_a_consistent_snapshot():
    """sigma circulating (every part spans a turn) or librating about the top of the well: no evaluation holds a swing
    in the well, so there is no T0_th and no pass."""
    record = pendulum_record()
    circulating = dataclasses.replace(record, angle_wrapped=np.mod(2 * np.pi * record.t_yr / 300.0, 2 * np.pi))
    context, _, prep = first_pass(circulating)
    assert preprocess.profile_period_floor(circulating, context, prep, SETTINGS) is None
    at_top = pendulum_record(centre_rad=np.pi)
    context, _, prep = first_pass(at_top)
    assert preprocess.profile_period_floor(at_top, context, prep, SETTINGS) is None


def test_eligibility_switches():
    series = pendulum_record()
    context, sampling, prep = first_pass(series)
    episodic = decide.AngleDecision(AngleStatus.INSUFFICIENT_TIME, AngleSubtype.STALL, 'G5')
    found = analyse_module.subfloor_eligible(series, context, sampling, prep, episodic, SETTINGS)
    assert found is not None and found.own_floor_yr == SETTINGS.mmr_libration_period_min_yr
    assert 2 * sampling.step_yr * SETTINGS.filter_min_libration_cutoffs <= found.floor_yr < found.own_floor_yr
    libration = decide.AngleDecision(AngleStatus.LIBRATION, AngleSubtype.STATIONARY, 'G2')
    assert analyse_module.subfloor_eligible(series, context, sampling, prep, libration, SETTINGS) is None
    off = settings_from_config(mmr_subfloor_pass=False)
    assert analyse_module.subfloor_eligible(series, context, sampling, prep, episodic, off) is None
    # an 8-yr step resolves the theory's floor (35 yr > 2 x 8 x 2 = 32 yr); at 20 yr the realisable bound (80 yr) binds
    # and its cutoff, 2 dt, cannot be filtered (P_c > 2 dt is required): no pass; at 40 yr it is above the floor
    fine = make_sampling(step_yr=8.0, verdict=sampling.verdict)
    assert analyse_module.subfloor_eligible(series, context, fine, prep, episodic, SETTINGS).floor_yr == pytest.approx(found.floor_yr)
    for step in (20.0, 40.0):
        coarse = make_sampling(step_yr=step, verdict=sampling.verdict)
        assert analyse_module.subfloor_eligible(series, context, coarse, prep, episodic, SETTINGS) is None


# --- the conditions of a kept pass (truth table) -----------------------------------------------------------------------


def _pass(f_pass: bool, centre_ok: bool, l4_pass: bool, period_yr: float = 105.0) -> models.MMRAnalysis:
    """A G1f pass whose decisive episode (60 deg smallest half-amplitude) has `period_yr`; F by direct averaging passes
    or fails, the theoretical centre is 10 deg or 70 deg from the observed ones, L4 passes or fails."""
    episode = make_episode(n_tp=5, episode_id=0, period_med_yr=period_yr, half_amp_min_rad=math.radians(60.0))
    strength = make(
        models.Strength,
        method=strength_module.METHOD_DIRECT,
        check=CheckValue.PASS if f_pass else CheckValue.FAIL,
        centre_offset_max_deg=10.0 if centre_ok else 70.0,
    )
    checks = {'check_mmr_a_response': CheckValue.PASS if l4_pass else CheckValue.FAIL}
    verdict = Verdict(ResonanceType.MMR, Status.RESONANT, AngleStatus.LIBRATION, checks=checks)
    return make(models.MMRAnalysis, geometry=make_geometry(runs=[episode]), strength=strength, verdict=verdict, decisive_episode_id=0)


@pytest.mark.parametrize('f_pass, centre_ok, l4_pass', list(itertools.product([True, False], repeat=3)))
def test_kept_only_when_f_centre_and_l4_all_pass(f_pass, centre_ok, l4_pass):
    ledger = Ledger.from_settings(SETTINGS)
    failed = analyse_module.subfloor_conditions(_pass(f_pass, centre_ok, l4_pass), 150.0, ledger)
    expected = [name for name, ok in (('F', f_pass), ('centre', centre_ok), ('L4', l4_pass)) if not ok]
    assert failed == expected
    assert len(ledger) == 4  # the cycle, F, centre and L4 comparisons, all recorded for the decision path


def test_a_pass_without_a_sub_floor_cycle_is_not_kept():
    """Everything confirmed, but the decisive cycle (155 yr, 2001 WK15's) is not below the floor: the pass found
    nothing the floor hid."""
    failed = analyse_module.subfloor_conditions(_pass(True, True, True, period_yr=155.0), 150.0, Ledger.from_settings(SETTINGS))
    assert failed == [analyse_module.SUBFLOOR_NO_CYCLE]


def test_eq_scale_strength_is_no_f_by_direct_averaging():
    analysis = _pass(True, True, True)
    analysis.strength = make(models.Strength, method=strength_module.METHOD_SCALE, check=CheckValue.PASS, centre_offset_max_deg=10.0)
    failed = analyse_module.subfloor_conditions(analysis, 150.0, Ledger.from_settings(SETTINGS))
    assert failed == [analyse_module.SUBFLOOR_NO_F, analyse_module.SUBFLOOR_NO_CENTRE]


def test_columns_without_a_pass():
    columns = analyse_module.subfloor_columns(None)
    assert math.isnan(columns['geo_subfloor_kept']) and math.isnan(columns['geo_subfloor_floor_yr'])
    assert columns['geo_subfloor_failed'] == ''


# --- whole analyses (slow) ---------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_two_body_pendulum_of_100_yr_is_recovered():
    """T0 = 100 yr, A = 60 deg about the theoretical centre: the 150-yr floor hides it (no libration), G1f recovers it
    at the period of a 60-deg pendulum, 107.3 yr (one 5-yr step of tolerance)."""
    series = pendulum_record()
    off = analyse_module.analyse_mmr(series, settings_from_config(mmr_subfloor_pass=False), scan_owner=False, scan_subresonances=False)
    assert off.verdict.angle_status != AngleStatus.LIBRATION
    kept = analyse_module.analyse_mmr(series, SETTINGS, scan_owner=False, scan_subresonances=False)
    assert (kept.verdict.status, kept.verdict.angle_status) == (Status.RESONANT, AngleStatus.LIBRATION)
    assert 'subfloor_pass' in kept.verdict.rules_fired
    expected = 100.0 * 2 * ellipk(math.sin(math.radians(30.0)) ** 2) / math.pi  # scipy's ellipk takes m = k^2
    assert kept.extra_columns['geo_subfloor_period_yr'] == pytest.approx(expected, abs=5.0)
    assert kept.extra_columns['geo_subfloor_kept'] == 1.0
    assert kept.prep.period_floor_basis == preprocess.SUBFLOOR_SOURCE
    assert kept.verdict.decision_path[0].name == analyse_module.SUBFLOOR_FLOOR


@pytest.mark.slow
def test_forced_three_body_wave_of_118_yr_is_not_recovered():
    analysis = analyse_module.analyse_mmr(forced_record(), SETTINGS, scan_owner=False, scan_subresonances=False)
    assert math.isnan(analysis.extra_columns['geo_subfloor_kept'])
    assert analysis.verdict.angle_status != AngleStatus.LIBRATION
    assert 'subfloor_pass' not in analysis.verdict.rules_fired
