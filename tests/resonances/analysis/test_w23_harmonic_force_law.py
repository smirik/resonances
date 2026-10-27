"""W23 CS12: the force law of L4 with N harmonics of the angle (`response.harmonic_law`).

sigma_dot = c (a - a*) and sigma_ddot = c a_dot = -V'(sigma), so a pendulum in the well V(sigma) = -w^2 (cos sigma +
q cos 2 sigma) has the force law a_dot = -(w^2 / c) (sin sigma + 2 q sin 2 sigma): one harmonic for q = 0, two for
q != 0, with |C_2| / |C_1| = 2 |q| exactly. The synthetic records below integrate that pendulum (scipy, not the
package) and add 30 % noise to a_dot; the closed forms they are checked against are the ratio 2|q|, the noise ceiling
R^2 = 1 / (1 + 0.3^2) = 0.917 and, for a law that changes sign between the halves, an out-of-sample R^2 of
1 - 4 = -3. The real records' numbers are those of the W23 design runs (design/cs12, variant 'ho'), not the
package's output: 2001 ND13 1V-3 harmonic R^2 0.646 (one harmonic 0.18), 33128 6N-7 refused on independent halves
(0.26), 187661 7N-5 and 496028 9E-41 never lifted. The benchmark records are skipped without the benchmark data.
"""

from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import analyse as analyse_module, decide, io
from resonances.resonance.analysis.mmr import response as R
from resonances.resonance.analysis.mmr.models import HarmonicLaw, MultipletFit, ResonanceSpec
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.ledger import Margin
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, CheckValue, LibrationOrigin, Reason, Status
from tests.resonances.analysis.builders import make, make_geometry, make_response

S = settings_from_config()
T = np.arange(0.0, 20000.0, 5.0)
W2 = (2 * np.pi / 1000) ** 2  # small-amplitude period 1000 yr for q = 0
NOISE = 0.3
NOISE_CEILING = 1 / (1 + NOISE**2)  # R^2 of an exact law under white noise of 0.3 x its spread


def _well(q, energy, seed=1):
    """(sigma, noisy a_dot, period) of the pendulum in V = -W2 (cos s + q cos 2s), started at s = 0 with energy
    `energy` x W2 above V(0); c = 10."""

    def force(x):
        return W2 * (np.sin(x) + 2 * q * np.sin(2 * x))

    sol = solve_ivp(
        lambda _, y: [y[1], -force(y[0])], (0, T[-1]), [0.0, 0.999 * np.sqrt(2 * energy * W2)], t_eval=T, rtol=1e-10, atol=1e-12
    )
    sigma, rate = sol.y
    adot = -force(sigma) / 10.0
    adot = adot + NOISE * np.std(adot) * np.random.default_rng(seed).standard_normal(len(adot))
    crossings = np.flatnonzero(np.diff(np.sign(rate)) != 0)
    return sigma, adot, 2 * float(np.median(np.diff(T[crossings])))


def _fit(sigma, period, spec=None):
    return R.ForceFit(T, [(T[0], T[-1])], S.mmr_force_window_cycles * period, None, spec, None, None, np.ptp(sigma) / 2, S)


def _single_r2(measurement):
    return min(x for x in R._halves(measurement.windows) if np.isfinite(x))


# --- the fits on closed forms -------------------------------------------------------------------------------------


def test_harmonic_fits_recover_an_exact_two_harmonic_law():
    angle = np.linspace(0, 40 * np.pi, 4000)
    t = np.arange(len(angle), dtype=float)
    y = 2.0 * np.sin(angle) - 3.0 * np.cos(2 * angle) + 0.5
    coef, r2, count = R.harmonic_fits(t, angle, y, [(0.0, 1999.0), (1000.0, 3999.0)], 2)
    assert np.allclose(coef, [[2.0, 0.0, 0.0, -3.0, 0.5]] * 2, atol=1e-9)
    assert np.allclose(r2, 1.0) and list(count) == [2000, 3000]
    # a window with fewer samples than 2N + 3 is no fit
    _, r2, _ = R.harmonic_fits(t, angle, y, [(0.0, 5.0)], 2)
    assert np.isnan(r2[0])


def test_one_harmonic_of_harmonic_fits_is_the_single_law():
    sigma, adot, period = _well(-1.0, 1.0)
    windows = R.sliding_windows(T, 3.5 * period, 0.5)
    a_sin, b_cos, offset, r2_single = R.sine_fits_in_windows(T, sigma, adot, windows)
    coef, r2, _ = R.harmonic_fits(T, sigma, adot, windows, 1)
    assert np.allclose(coef[:, 0], a_sin) and np.allclose(coef[:, 1], b_cos) and np.allclose(coef[:, 2], offset)
    assert np.allclose(r2, r2_single)


def test_lower_half_median_reads_the_windows_in_time_order():
    # starts out of order: in time the R^2 are (0.1, 0.9 | 0.5, 0.7): medians 0.5 and 0.6
    assert R.lower_half_median([30.0, 0.0, 10.0, 20.0], np.array([0.7, 0.1, 0.9, 0.5])) == pytest.approx(0.5)
    assert np.isnan(R.lower_half_median([], np.array([])))


def test_shifted_rates_use_quarter_period_lags():
    t, adot = np.arange(101.0), np.arange(101.0)
    moved = list(R.shifted_rates(t, adot, 10.0, 8))
    # half the record 50 yr: whole periods 1..4 (4.75 P <= 50), round(linspace(1, 4, 8)) = 1 1 2 2 3 3 4 4
    lags = [12, 18, 22, 28, 32, 38, 42, 48]  # (m + 1/4) P and (m + 3/4) P, Python's round
    for j, (series, lag) in enumerate(zip(moved, lags)):
        if j % 2 == 0:
            assert np.all(np.isnan(series[:lag])) and np.array_equal(series[lag:], adot[: len(adot) - lag])
        else:
            assert np.array_equal(series[: len(adot) - lag], adot[lag:]) and np.all(np.isnan(series[len(adot) - lag :]))
    assert list(R.shifted_rates(np.arange(30.0), np.arange(30.0), 10.0, 8)) == []  # T / 2 < 1.75 P


def test_independent_halves_catch_a_law_that_changes_between_the_halves():
    t = np.arange(0.0, 10000.0, 1.0)
    sigma = 2.5 * np.sin(2 * np.pi * t / 500)
    steady = np.sin(sigma)
    flipped = np.where(t < 5000, 1.0, -1.0) * np.sin(sigma)
    windows = [(0.0, 4999.0), (5000.0, 9999.0)]
    assert R.independent_halves_r2(t, sigma, steady, windows, 2) == pytest.approx(1.0)
    # predicting -sin with sin leaves 2 sin: R^2 = 1 - 4 = -3
    assert R.independent_halves_r2(t, sigma, flipped, windows, 2) == pytest.approx(-3.0, abs=0.01)


def test_independent_halves_need_both_directions():
    """A target without spread on the angles one half visited leaves that direction unmeasured: nan, in either time
    order (Codex review S5: min() of a finite and a nan score kept the finite one)."""
    t = np.arange(0.0, 10000.0, 1.0)
    sigma = 2.5 * np.sin(2 * np.pi * t / 500)
    flat_late = np.where(t < 5000, np.sin(sigma), 0.0)
    windows = [(0.0, 4999.0), (5000.0, 9999.0)]
    assert np.isnan(R.independent_halves_r2(t, sigma, flat_late, windows, 2))
    assert np.isnan(R.independent_halves_r2(t, sigma, flat_late[::-1].copy(), windows, 2))


# --- synthetic wells through L4 -----------------------------------------------------------------------------------


@pytest.mark.parametrize('energy', [0.3, 1.0, 1.7])
def test_a_double_well_lifts_the_force_law(energy):
    """V = -cos s + cos 2s (q = -1), swinging over both wells: one harmonic fails, two give the noise ceiling."""
    sigma, adot, period = _well(-1.0, energy)
    measurement = _fit(sigma, period).measure(sigma, adot, [])
    law = measurement.harmonic
    assert _single_r2(measurement) < S.mmr_force_r2_fail  # one harmonic explains < 0.3
    assert law.taken and law.reason == '' and law.n_harmonics == 2
    assert law.r2 >= 0.9 and law.r2 == pytest.approx(NOISE_CEILING, abs=0.02)
    assert law.c2_over_c1 == pytest.approx(2.0, abs=0.1)  # 2 |q|
    assert law.coherence >= 0.99 and law.null_max <= 0.1 and law.oos_r2 >= 0.9
    assert measurement.check == (CheckValue.PASS, False, R.LAW_HARMONIC)
    assert measurement.law_r2 == law.r2


def test_a_shallow_double_well_is_lifted_and_a_flat_bottom_is_not_needed():
    """q = -0.5 (barrier at 0, minima at +-60 deg): taken, |C2|/|C1| = 1. q = -0.25 (V''(0) = 0, a flat bottom): the
    two-harmonic law fits it too, but one harmonic already passes with its full margin, so L4 does not consult it."""
    sigma, adot, period = _well(-0.5, 1.0)
    measurement = _fit(sigma, period).measure(sigma, adot, [])
    assert measurement.harmonic.taken and measurement.harmonic.c2_over_c1 == pytest.approx(1.0, abs=0.05)
    assert measurement.check == (CheckValue.PASS, False, R.LAW_HARMONIC)

    sigma, adot, period = _well(-0.25, 1.0)
    fit = _fit(sigma, period)
    measurement = fit.measure(sigma, adot, [])
    assert measurement.check == (CheckValue.PASS, False, R.LAW_SINGLE) and measurement.harmonic is None
    direct = R.harmonic_law(fit, sigma, adot, [], _single_r2(measurement))
    assert direct.taken and direct.c2_over_c1 == pytest.approx(0.5, abs=0.05)


@pytest.mark.parametrize('energy', [0.3, 1.0, 1.9])  # ranges 91, 180 and 307 deg
def test_an_exact_one_harmonic_pendulum_gains_nothing(energy):
    sigma, adot, period = _well(0.0, energy)
    fit = _fit(sigma, period)
    measurement = fit.measure(sigma, adot, [])
    assert measurement.check == (CheckValue.PASS, False, R.LAW_SINGLE) and measurement.harmonic is None
    law = R.harmonic_law(fit, sigma, adot, [], _single_r2(measurement))
    assert not law.taken and law.reason == R.HARMONIC_NO_GAIN and abs(law.gain) <= 0.001


def test_a_kinematic_linear_swing_of_170_deg_is_lifted_thin():
    """a_dot = -(sigma - sigma0) is no pendulum, but sin sigma + sin 2 sigma fits a line better than sin sigma alone
    (thin single pass 0.52 -> thin harmonic pass 0.69): CS12 opens no new channel, and L4 is never sufficient alone
    (the kinematic tests, R0 and the strength decide the origin)."""
    sigma = np.radians(170) * np.sin(2 * np.pi * T / 1000)
    adot = -sigma + NOISE * np.std(sigma) * np.random.default_rng(1).standard_normal(len(T))
    measurement = _fit(sigma, 1000.0).measure(sigma, adot, [])
    assert measurement.harmonic.taken
    assert measurement.check == (CheckValue.PASS, True, R.LAW_HARMONIC)


def test_the_law_is_not_consulted_for_a_coorbital_angle_nor_when_off():
    sigma, adot, period = _well(-1.0, 1.0)
    measurement = _fit(sigma, period, ResonanceSpec.from_key('1J-1+0+0')).measure(sigma, adot, [])
    assert measurement.harmonic is None and measurement.check[0] == CheckValue.FAIL
    off = R.ForceFit(T, [(T[0], T[-1])], 3.5 * period, None, None, None, None, np.ptp(sigma) / 2, replace(S, mmr_force_harmonics_max=1))
    measurement = off.measure(sigma, adot, [])
    assert measurement.harmonic is None and measurement.check[0] == CheckValue.FAIL
    # inside the null test the law is not consulted either
    assert _fit(sigma, period).measure(sigma, adot, [], harmonic=False).harmonic is None


def test_a_shuffled_rate_is_never_lifted():
    sigma, adot, period = _well(-1.0, 1.0)
    shuffled = np.random.default_rng(2).permutation(adot)
    measurement = _fit(sigma, period).measure(sigma, shuffled, [])
    assert measurement.check[0] == CheckValue.FAIL and not measurement.harmonic.taken


# --- the tree -----------------------------------------------------------------------------------------------------


def _response(law, force_law):
    """A response whose one-harmonic halves (0.1, 0.2) and multiplet (0.3) all fail MMR_FORCE_R2_FAIL."""
    multiplet = make(MultipletFit, r2=0.3)
    return make_response(force_harmonic=law, force_law=force_law, force_r2_h1=0.1, force_r2_h2=0.2, multiplet=multiplet)


def test_k2_every_law_fails_reads_the_taken_harmonic_law():
    taken = HarmonicLaw(2, 0.65, 0.18, 2, coherence=0.99, null_max=0.1, oos_r2=0.6, c2_over_c1=1.9, taken=True)
    refused = replace(taken, taken=False, reason=R.HARMONIC_HALVES)
    assert not decide.force_law_fails(_response(taken, R.LAW_HARMONIC), S)
    assert [label for label, _, _ in decide.force_laws(_response(taken, R.LAW_HARMONIC))][-1] == '2 harmonics'
    # a refused fit is no law, whatever its R^2: the halves and the multiplet decide
    labels = [label for label, _, _ in decide.force_laws(_response(refused, R.LAW_SINGLE))]
    assert labels == ['half 1', 'half 2', 'multiplet'] and decide.force_law_fails(_response(refused, R.LAW_SINGLE), S)


def test_the_ledger_records_the_guards_of_the_harmonic_law():
    law = HarmonicLaw(2, 0.646, 0.18, 2, coherence=0.995, null_max=0.095, oos_r2=0.61, c2_over_c1=1.9, taken=True)
    ledger = Ledger.from_settings(S)
    decide._force_law_stability(_response(law, R.LAW_HARMONIC), S, ledger)
    entries = {entry.name: entry for entry in ledger}
    assert entries[decide.HARMONIC_NULL].value == pytest.approx(0.646 - 0.095) and entries[decide.HARMONIC_NULL].result
    assert entries[decide.HARMONIC_HALVES].value == pytest.approx(0.61) and entries[decide.HARMONIC_HALVES].result
    assert not entries[f'{decide.HARMONIC_LAW} R2'].result  # 0.646 < MMR_FORCE_R2_FULL: a thin pass
    assert all(entry.column.startswith('rsp_force_harm_') for entry in ledger)


def test_the_l4_margin_reads_the_law_that_took_it():
    law = HarmonicLaw(2, 0.646, 0.18, 2, coherence=0.995, null_max=0.095, oos_r2=0.61, c2_over_c1=1.9, taken=True)
    response = replace(_response(law, R.LAW_HARMONIC), a_response=CheckValue.PASS, a_response_thin=True)
    decision = decide.StatusDecision(Status.TRANSIENT, Reason.DYNAMIC_EPISODE, LibrationOrigin.DYNAMIC, '', '', rule='6')
    margin = decide._l4(response, decision)
    assert margin.margin == Margin.THIN and margin.value == pytest.approx(0.646)
    margin = decide._l4(replace(response, force_harmonic=None, force_law=R.LAW_SINGLE), decision)
    assert margin.value == pytest.approx(0.1)  # the smaller one-harmonic half
    # the one-harmonic halves are no convention the harmonic verdict rests on: never a thin U margin
    ledger = Ledger.from_settings(S)
    halves = replace(response, force_r2_h1=0.69, force_r2_h2=0.69)
    episodic = decide.AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, 'G2', 0)
    decide._force_and_potential(episodic, make_geometry(), halves, None, S, ledger)
    assert not any('force law R2 (half' in item for item in ledger.thin_margins())


# --- real records -------------------------------------------------------------------------------------------------

ND13 = 'mmr2__2001_ND13__1V-3+0+2'


@lru_cache(maxsize=None)
def _analysis(example):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    return analyse_module.analyse_mmr(io.load_series(example), S)


def _numbers(example):
    analysis = _analysis(example)
    return analysis.response.to_columns(), analysis.verdict


def test_nd13_libration_in_a_two_lobed_well_is_a_dynamic_episode():
    columns, verdict = _numbers(ND13)
    assert (str(verdict.status), str(verdict.angle_status), str(verdict.confidence)) == ('transient', 'episodic', 'medium')
    assert (str(verdict.reason), str(verdict.libration_origin)) == ('dynamic_episode', 'dynamic')
    assert columns['rsp_force_law'] == R.LAW_HARMONIC and columns['rsp_force_harm_N'] == 2
    assert columns['rsp_force_harm_R2'] == pytest.approx(0.646, abs=0.02)
    assert columns['rsp_force_R2_h1'] == pytest.approx(0.18, abs=0.02)  # one harmonic
    assert columns['rsp_force_harm_coherence'] >= 0.99 and columns['rsp_force_harm_null_max'] == pytest.approx(0.095, abs=0.02)
    assert columns['rsp_force_harm_c2_over_c1'] == pytest.approx(1.9, abs=0.1)


def test_33128_is_refused_on_independent_halves():
    columns, verdict = _numbers('mmr2__33128__6N-7+0+1')
    assert columns['rsp_force_harm_reason'] == R.HARMONIC_HALVES
    assert columns['rsp_force_harm_oos'] == pytest.approx(0.26, abs=0.02)
    assert columns['rsp_force_law'] != R.LAW_HARMONIC
    assert (str(verdict.status), str(verdict.confidence)) == ('transient', 'high')


@pytest.mark.parametrize('example', ['mmr2__187661__7N-5+0-2', 'mmr2__496028__9E-41+0+32'])
def test_circulations_are_not_lifted(example):
    columns, verdict = _numbers(example)
    assert columns['rsp_force_law'] != R.LAW_HARMONIC and columns['rsp_force_harm_reason'] == R.HARMONIC_INCOHERENT
    assert str(verdict.status) == 'non_resonant' and str(verdict.angle_status) == 'circulation'
