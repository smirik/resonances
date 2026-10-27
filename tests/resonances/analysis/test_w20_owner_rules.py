"""W20 CS3: the owner's rules without the benchmark (R5 §1, R1 §2): the Psi lever, the phase route, slips, the energy
test of a second capture, rate admissibility, the potential route on a synthetic record, the identity text, the
intervals and their cover (F7), and the decision on an owner outside sigma's episode.

Every expected value is computed independently of the package: closed forms (sinc), a pendulum integrated with
scipy, the averaged potential differentiated by np.gradient instead of the package's FFT."""

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from resonances.data import const
from resonances.resonance.analysis import filtering
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import averaged_potential as averaging
from resonances.resonance.analysis.mmr import decide, owner, owner_potential
from resonances.resonance.analysis.mmr.coorbital import Orbit
from resonances.resonance.analysis.mmr.models import (
    MMRSeries,
    OwnerCandidate,
    OwnerResult,
    PlanetSeries,
    Regime,
    ResonanceSpec,
    intervals_text,
)
from resonances.resonance.analysis.mmr.response import sine_fits_in_windows, sliding_windows
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import CheckValue, Flag

S = settings_from_config()
TWO_PI = 2 * math.pi
NAN = float('nan')


# --------------------------------------------------------------------------------------------- Psi and the lever


def test_psi_combination_removes_the_body_lambda():
    # 385266: sigma = 16N-27+0+11, owner 3N-5+0+2: gcd(27, 5) = 1, u = 5, v = 27 (signs flipped together)
    assert owner.psi_combination(ResonanceSpec.from_key('16N-27+0+11'), ResonanceSpec.from_key('3N-5+0+2')) == (5, 27)
    # GK3: 3M+5S-3+0+0-5 and 10E-17+0+7: (17, 3); 78085 5J-4S-1 and 2E-7+0+5: (7, 1)
    assert owner.psi_combination(ResonanceSpec.from_key('3M+5S-3+0+0-5'), ResonanceSpec.from_key('10E-17+0+7')) == (17, 3)
    assert owner.psi_combination(ResonanceSpec.from_key('5J-4S-1+0+0+0'), ResonanceSpec.from_key('2E-7+0+5')) == (7, 1)
    # a common factor: k = -4, k_o = -6 -> d = 2, (3, 2)
    assert owner.psi_combination(ResonanceSpec.from_key('3J-4+0+1'), ResonanceSpec.from_key('5J-6+0+1')) == (3, 2)
    for sigma, phi in (('16N-27+0+11', '3N-5+0+2'), ('3M+5S-3+0+0-5', '10E-17+0+7')):
        assert 'lambda' not in owner.psi_terms(ResonanceSpec.from_key(sigma), ResonanceSpec.from_key(phi))


@pytest.mark.parametrize('delta, expected', [(math.pi, 2 / math.pi), (TWO_PI, 0.0), (0.0, 1.0), (1.3, math.sin(0.65) / 0.65)])
def test_lever_of_a_uniform_drift_is_the_sinc(delta, expected):
    # Psi_w / v spread uniformly over [0, delta]: |<exp(i x)>| -> |sin(delta / 2) / (delta / 2)| (continuous limit)
    v, u = 3, 17
    psi = v * np.linspace(0.0, delta, 20001)
    lever_owner, _ = owner.phase_leverage(psi, u, v)
    assert lever_owner == pytest.approx(abs(expected), abs=1e-3)
    # sigma's lever reads Psi / u
    _, lever_sigma = owner.phase_leverage(u * np.linspace(0.0, delta, 20001), u, v)
    assert lever_sigma == pytest.approx(abs(expected), abs=1e-3)


def test_lever_of_constant_phases_is_one():
    rng = np.random.default_rng(7)
    constant = np.full(25, rng.uniform(-50, 50))
    assert owner.phase_leverage(constant, 17, 3) == pytest.approx((1.0, 1.0))
    assert np.isnan(owner.phase_leverage(np.array([]), 17, 3)[0])
    assert np.isnan(owner.phase_leverage(constant, 0, 0)[0])


def test_window_means_by_hand():
    t = np.arange(10.0)
    values = t**2
    means = owner.window_means(t, values, [(0, 2), (3, 3.5), (8, 20), (11, 12)])
    assert means[0] == pytest.approx((0 + 1 + 4) / 3) and means[1] == pytest.approx(9.0)
    assert means[2] == pytest.approx((64 + 81) / 2) and np.isnan(means[3])


def _phase_numbers(t, angle_b, sigma, adot, length):
    """(C_B, C_sigma, median R^2 of B, of sigma) of a_dot on sin/cos of each angle in sliding windows (as the owner fit)."""
    windows = sliding_windows(t, length, 0.5)
    out = []
    for angle in (angle_b, sigma):
        a_sin, b_cos, _, r2 = sine_fits_in_windows(t, angle, adot, windows)
        out.append((filtering.circular_resultant(np.arctan2(-b_cos, a_sin)), float(np.median(r2))))
    return out[0][0], out[1][0], out[0][1], out[1][1], windows


def _synthetic(drive_sigma: bool, psi_period_yr):
    """R5 §7.2: u = 17, v = 3, Psi = 2 pi t / T (or 0), 20 kyr, step 2 yr, 5 % noise on a_dot. The owner drives:
    phi_o = pi + 0.6 sin(2 pi t / 225) about a fixed centre, sigma = (v/u) phi_o + Psi / u, a_dot ~ sin(phi_o - pi).
    sigma drives (the mirror): sigma = 1 + (v/u) 0.6 sin(2 pi t / 225) about a fixed centre, phi_o = (u/v)(sigma - Psi/u)
    by the identity (its centre moves with -Psi / v), a_dot ~ sin(sigma - 1)."""
    u, v = 17, 3
    t = np.arange(0.0, 20000.0, 2.0)
    psi = TWO_PI * t / psi_period_yr if psi_period_yr else np.zeros_like(t)
    wave = 0.6 * np.sin(TWO_PI * t / 225.0)
    if drive_sigma:
        sigma = 1.0 + (v / u) * wave
        phi = (u / v) * (sigma - psi / u)
        clean = np.sin(sigma - 1.0)
    else:
        phi = math.pi + wave
        sigma = (v / u) * phi + psi / u
        clean = np.sin(phi - math.pi)
    adot = clean + 0.05 * np.std(clean) * np.random.default_rng(11).normal(size=len(t))
    c_b, c_s, r2_b, r2_s, windows = _phase_numbers(t, phi, sigma, adot, 3.5 * 225.0)
    lever = owner.phase_leverage(owner.window_means(t, u * sigma - v * phi, windows), u, v)
    candidate = OwnerCandidate(
        spec=ResonanceSpec.from_key('10E-17+0+7'),
        rate_rad_yr=0.0,
        r2_median=r2_b,
        r2_windows=(r2_b,) * 3,
        phi0_coherence=c_b,
        psi_share=0.0,
        force_au_yr=1.0,
        force_theory_au_yr=1.0,
        strength_ratio=1.0,
        rank=1,
        accepted=False,
        reason='',
        lever_owner=lever[0],
        lever_sigma=lever[1],
    )
    return candidate, c_s, r2_s


def test_phase_route_accepts_the_owner_the_old_rule_rejects():
    # Psi turns 2.1 times over the window centres (19.2 kyr): Psi / u spans 0.79 rad (sigma's coherence under the owner
    # sinc(0.39) = 0.97), Psi / v 4.47 rad (the owner's coherence could not exceed |sinc(2.24)| = 0.35 if sigma drove a).
    # R5's 1640-yr Psi gives Psi / u = 4.5 rad and C_sigma = 0.35: the old rule accepts that one too, so it cannot show
    # the difference.
    candidate, c_sigma, r2_sigma = _synthetic(False, 9000.0)
    assert c_sigma > 0.8 and candidate.phi0_coherence > 0.95
    span = TWO_PI * (20000.0 - 2.0 - 3.5 * 225.0) / 9000.0 / 3  # Psi / v across the window centres (L/2 .. T - L/2)
    assert candidate.lever_owner == pytest.approx(abs(math.sin(span / 2) / (span / 2)), abs=0.05)
    assert not candidate.phi0_coherence >= c_sigma + S.mmr_owner_phi0_coh_gain  # the old phase route rejects B
    old = replace(candidate, lever_owner=NAN, lever_sigma=NAN)
    assert owner.better_than_sigma(old, r2_sigma, c_sigma, NAN, S) == ''
    assert owner.better_than_sigma(candidate, r2_sigma, c_sigma, NAN, S) == owner.GAIN_PHASE
    assert owner.accept(candidate, r2_sigma, S, sigma_coherence=c_sigma) == (True, 'accepted:phase')


def test_mirror_case_sigma_drives_and_the_owner_is_not_accepted():
    candidate, c_sigma, r2_sigma = _synthetic(True, 9000.0)
    assert c_sigma > 0.9  # sigma's own phase is fixed
    assert candidate.phi0_coherence < candidate.lever_owner + S.mmr_owner_phi0_coh_gain
    assert owner.better_than_sigma(candidate, r2_sigma, c_sigma, NAN, S) == ''
    assert owner.accept(candidate, r2_sigma, S, sigma_coherence=c_sigma)[0] is False


def test_constant_psi_leaves_the_phase_unresolved():
    candidate, c_sigma, r2_sigma = _synthetic(False, 0.0)
    assert candidate.lever_owner == pytest.approx(1.0)
    assert owner.better_than_sigma(candidate, r2_sigma, c_sigma, NAN, S) == owner.PHASE_UNRESOLVED
    assert owner.accept(candidate, r2_sigma, S, sigma_coherence=c_sigma) == (False, owner.REASON_PHASE_UNRESOLVED)


def test_psi_has_no_jump_across_a_slip_and_the_lever_corrects_sigma():
    u, v = 17, 3
    t = np.arange(0.0, 1000.0, 1.0)
    phi = 0.3 * np.sin(TWO_PI * t / 50.0) + TWO_PI * (t >= 500.0)  # one slip of the owner at t* = 500
    psi_true = 0.2 + 1e-3 * t
    sigma = (v / u) * phi + psi_true / u
    assert np.max(np.abs(np.diff(sigma))) > 0.9 * TWO_PI * v / u  # sigma jumps by 2 pi v / u at the slip
    psi = u * sigma - v * phi
    assert np.max(np.abs(np.diff(psi))) < 0.01 and np.allclose(psi, psi_true)
    # windows 0-1 and 3-4 librate, 2 holds the slip: the corrected Psi of the windows after the gap gains 2 pi v
    windows = [(0, 199), (200, 399), (400, 599), (600, 799), (800, 999)]
    psi_means, phi_means = owner.window_means(t, psi, windows), owner.window_means(t, phi, windows)
    index = np.array([0, 1, 3, 4])
    corrected = owner.slip_corrected(psi_means, phi_means, index, v)
    assert np.allclose(corrected - psi_means[index], [0.0, 0.0, TWO_PI * v, TWO_PI * v])
    # sigma_w - corrected_w / u = (v/u) (phi_w - 2 pi N_w): sigma follows the owner's angle on its libration branch
    sigma_means = owner.window_means(t, sigma, windows)[index]
    branch = phi_means[index] - TWO_PI * np.array([0, 0, 1, 1])
    assert np.allclose(sigma_means - corrected / u, (v / u) * branch)


# --------------------------------------------------------------------------------------------- rate admissibility


def test_rate_admissibility_by_hand():
    # P = 181 yr, f = 2: the band [90.5, 362] yr
    assert owner.rate_admissible(1.47, 181.0, 2.0) is False  # 2001 PG14 27E-47 in the decisive episode (R6)
    assert owner.rate_admissible(200.0, 181.0, 2.0) is True
    assert owner.rate_admissible(90.5, 181.0, 2.0) is True and owner.rate_admissible(362.0, 181.0, 2.0) is True
    assert owner.rate_admissible(362.5, 181.0, 2.0) is False
    assert owner.rate_admissible(NAN, 181.0, 2.0) is True  # not measured: no reason to reject
    candidate = OwnerCandidate(
        spec=ResonanceSpec.from_key('27E-47+0+20'),
        rate_rad_yr=4.27,
        r2_median=0.99,
        r2_windows=(0.99,),
        phi0_coherence=0.99,
        psi_share=0.0,
        force_au_yr=1.0,
        force_theory_au_yr=1.0,
        strength_ratio=1.0,
        rank=1,
        accepted=False,
        reason='',
        rate_admissible=False,
    )
    assert owner.accept(candidate, 0.0, S) == (False, owner.REASON_RATE)


# --------------------------------------------------------------------------------------------- the energy test


def test_energy_integral_of_a_pendulum_is_a_function_of_x():
    # x'' = -w^2 sin x, a wide libration (x0 = 2.6 rad: x sweeps 298 deg) with a slow damping of its energy
    w, damping = TWO_PI / 1800.0, 2e-6

    def rhs(_, y):
        return [y[1], -(w**2) * math.sin(y[0]) - damping * y[1]]

    t = np.arange(0.0, 30000.0, 2.0)
    x = solve_ivp(rhs, (0, t[-1]), [2.6, 0.0], t_eval=t, rtol=1e-10, atol=1e-12).y[0]
    capture = owner.second_capture(t, x, np.zeros_like(t), (1, 1), (0.0, t[-1]), 20.0, 1.0, S)
    assert math.degrees(capture.x_sweep_rad) >= S.mmr_owner_capture_sweep_deg
    assert capture.energy_r2 >= 0.9 and capture.verdict == owner.CAPTURE_UNRESOLVED


def test_energy_of_a_random_walk_rate_is_not_a_function_of_x():
    # x = a steady rate with a random-walk modulation of the rate: x_dot^2 is no function of x
    rng = np.random.default_rng(5)
    t = np.arange(0.0, 30000.0, 2.0)
    walk = np.cumsum(rng.normal(size=len(t)))
    walk = filtering.lowpass(t, walk, 600.0, 2)
    rate = (TWO_PI / 3000.0) * (1.0 + 0.5 * walk / np.max(np.abs(walk)))
    x = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))])
    capture = owner.second_capture(t, x, np.zeros_like(t), (1, 1), (0.0, t[-1]), 20.0, 1.0, S)
    assert capture.energy_r2 < 0.3 and math.degrees(capture.x_sweep_rad) > 360.0
    assert capture.verdict == owner.CAPTURE_EXCLUDED
    # psi_turns: the net turns of Psi on the trimmed span; lambda_ad = |<Psi_dot / v>| / omega0
    assert capture.psi_turns == pytest.approx((x[-1] - x[0]) / TWO_PI, rel=0.1)
    assert capture.adiabatic == pytest.approx(np.mean(rate), rel=0.1)


def _forced_pendulum():
    """x'' = -w^2 sin x (w = 2 pi / 1800 yr, x0 = 2.6 rad: a libration sweeping 298 deg) on [0, 12] kyr, driven round by
    a force 1.5 w^2 + 2.2 w^2 sin(2 pi t / 2300) on (12, 18) kyr (59 turns), then free again: a circulation of constant
    energy after 18 kyr. On each of the two free stretches x_dot^2 is a function of x; across the forced gap it is not."""
    w = TWO_PI / 1800.0

    def rhs(t, y):
        forced = 1.5 * w**2 + 2.2 * w**2 * math.sin(TWO_PI * t / 2300.0) if 12000.0 < t < 18000.0 else 0.0
        return [y[1], -(w**2) * math.sin(y[0]) + forced]

    t = np.arange(0.0, 30000.0, 2.0)
    return t, solve_ivp(rhs, (0, t[-1]), [2.6, 0.0], t_eval=t, rtol=1e-10, atol=1e-12, max_step=5.0).y[0]


def test_second_capture_is_tested_on_each_owner_block():
    """FIX1 (Codex A #6): blocks [0, 12] and [18, 30] kyr with the gap (12, 18) kyr where the owner does not hold phi_o.
    Each block alone conserves the energy (R^2 > 0.95: unresolved); their hull takes in the forced gap (R^2 ~ 0.02,
    excluded), which the old test on the hull reported."""
    t, x = _forced_pendulum()
    phi = np.zeros_like(t)
    blocks = [(0.0, 12000.0), (18000.0, float(t[-1]))]
    hull = owner.second_capture(t, x, phi, (1, 1), (0.0, float(t[-1])), 20.0, 1.0, S)
    assert hull.verdict == owner.CAPTURE_EXCLUDED and hull.energy_r2 < 0.1
    for block in blocks:
        alone = owner.second_capture(t, x, phi, (1, 1), block, 20.0, 1.0, S)
        assert alone.verdict == owner.CAPTURE_UNRESOLVED and alone.energy_r2 > 0.95
    capture = owner.second_capture_blocks(t, x, phi, (1, 1), blocks, 20.0, 1.0, S)
    assert capture.verdict == owner.CAPTURE_UNRESOLVED and (capture.t_start_yr, capture.t_end_yr) == blocks[0]


def test_second_capture_is_excluded_only_when_every_block_excludes():
    # the random-walk rate of the test above in both blocks: excluded, shown with the larger (weaker) R^2 of the two
    rng = np.random.default_rng(5)
    t = np.arange(0.0, 30000.0, 2.0)
    walk = filtering.lowpass(t, np.cumsum(rng.normal(size=len(t))), 600.0, 2)
    rate = (TWO_PI / 3000.0) * (1.0 + 0.5 * walk / np.max(np.abs(walk)))
    x = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))])
    blocks = [(0.0, 12000.0), (18000.0, float(t[-1]))]
    single = [owner.second_capture(t, x, np.zeros_like(t), (1, 1), block, 20.0, 1.0, S) for block in blocks]
    assert all(c.verdict == owner.CAPTURE_EXCLUDED for c in single)
    capture = owner.second_capture_blocks(t, x, np.zeros_like(t), (1, 1), blocks, 20.0, 1.0, S)
    assert capture.verdict == owner.CAPTURE_EXCLUDED and capture.energy_r2 == max(c.energy_r2 for c in single)
    # one block of the conserved libration instead: no longer excluded
    t2, libration = _forced_pendulum()
    mixed = np.where(t < 15000.0, libration, x)
    assert owner.second_capture_blocks(t, mixed, np.zeros_like(t), (1, 1), blocks, 20.0, 1.0, S).verdict == owner.CAPTURE_UNRESOLVED


def test_an_unmeasured_block_leaves_the_second_capture_open():
    """Review B #3: the random-walk block excludes, a block of 4 samples (8 yr) next to it cannot be measured (energy
    R^2 nan): capture was never tested there, so the result is unresolved, with the unmeasured block's numbers; no
    block at all is unresolved too."""
    rng = np.random.default_rng(5)
    t = np.arange(0.0, 30000.0, 2.0)
    walk = filtering.lowpass(t, np.cumsum(rng.normal(size=len(t))), 600.0, 2)
    rate = (TWO_PI / 3000.0) * (1.0 + 0.5 * walk / np.max(np.abs(walk)))
    x = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))])
    blocks = [(0.0, 12000.0), (20000.0, 20006.0)]
    single = [owner.second_capture(t, x, np.zeros_like(t), (1, 1), block, 20.0, 1.0, S) for block in blocks]
    assert single[0].verdict == owner.CAPTURE_EXCLUDED and np.isnan(single[1].energy_r2)
    capture = owner.second_capture_blocks(t, x, np.zeros_like(t), (1, 1), blocks, 20.0, 1.0, S)
    assert capture.verdict == owner.CAPTURE_UNRESOLVED and (capture.t_start_yr, capture.t_end_yr) == blocks[1]
    empty = owner.second_capture_blocks(t, x, np.zeros_like(t), (1, 1), [], 20.0, 1.0, S)
    assert empty.verdict == owner.CAPTURE_UNRESOLVED and np.isnan(empty.energy_r2)


# --------------------------------------------------------------------------------------------- the potential route


def _potential_record():
    """A body in a forced wave of the 2:1 with Jupiter: fixed elements, phi_o = 1J-2+0+1 circulating at 2 pi / 500 yr,
    a_dot = (2 k / (n a)) G m_J d<D>/dphi (phi_o(t)) with d<D>/dphi from np.gradient of the averaged profile."""
    spec = ResonanceSpec.from_key('1J-2+0+1')
    t = np.arange(0.0, 20000.0, 1.0)
    a0, e, inc, node, varpi = 3.25, 0.12, 0.08, 1.1, 0.4
    a_j, e_j, inc_j, node_j, varpi_j = 5.2, 0.048, 0.02, 1.75, 0.25
    mu = const.PLANETS_MASS['Jupiter']
    n_j = TWO_PI / 11.86
    lambda_j = 0.3 + n_j * t
    phi = 0.7 + TWO_PI * t / 500.0
    lam = (phi - lambda_j - varpi) / -2.0  # phi = lambda_J - 2 lambda + varpi
    body = averaging.Ellipse(a0, Orbit(e, inc, varpi - node, node))
    planet = averaging.Ellipse(a_j, Orbit(e_j, inc_j, varpi_j - node_j, node_j))
    grid, values = averaging.averaged_profile(body, planet, -2, 1, (-2 + 1) * varpi, 1440)
    slope = np.gradient(np.concatenate([values[-2:], values, values[:2]]), np.concatenate([grid[-2:] - TWO_PI, grid, grid[:2] + TWO_PI]))[
        2:-2
    ]
    n = math.sqrt(4 * math.pi**2 / a0**3)
    coefficient = 2 * -2 / (n * a0) * 4 * math.pi**2 * mu
    adot = coefficient * np.interp(np.mod(phi, TWO_PI), grid, slope, period=TWO_PI)
    wave = np.concatenate([[0.0], np.cumsum(0.5 * (adot[1:] + adot[:-1]))])
    a = a0 + wave - np.median(wave)  # the window medians of a stay at a0, where the profile was computed
    ones = np.ones_like(t)
    jupiter = PlanetSeries('Jupiter', t, a_j * ones, e_j * ones, inc_j * ones, node_j * ones, varpi_j * ones, np.mod(lambda_j, TWO_PI))
    series = MMRSeries(
        t_yr=t,
        a=a,
        e=e * ones,
        inc=inc * ones,
        Omega=node * ones,
        omega=(varpi - node) * ones,
        varpi=varpi * ones,
        angle_wrapped=np.mod(phi, TWO_PI),
        resonance=ResonanceSpec.from_key('3J-7+0+4'),
        regime=Regime(step_yr=1.0, span_yr=t[-1], n_rows=len(t)),
        planets={'Jupiter': jupiter},
        mean_longitude=np.mod(lam, TWO_PI),
    )
    expected = owner_potential.sine_projection(grid, coefficient * slope)
    return spec, series, phi, adot, mu, expected


def test_potential_route_recovers_the_force_it_was_built_from():
    spec, series, phi, adot, mu, (k_expected, phi0_expected) = _potential_record()
    t = series.t_yr
    period, factor = 500.0, S.mmr_owner_band_factor
    mask = filtering.window_mask(t, (factor * period, t[-1] - factor * period))
    fit_mask = mask & (np.arange(len(t)) % 5 == 0)
    a_band = filtering.band_pass(t, series.a, period, factor, S.filter_order)
    frame = SimpleNamespace(
        t_yr=t, fit_mask=fit_mask, window_yr=3.5 * period, period_yr=period, adot_band=filtering.time_derivative(t, a_band)
    )
    starts = [2000.0, 6000.0, 10000.0, 14000.0]
    phi_fit = filtering.lowpass(t, phi, 30.0, S.filter_order)[fit_mask]
    context = SimpleNamespace(planet_mass={'Jupiter': mu})
    fit = owner_potential.owner_potential(spec, starts, frame, np.mod(phi, TWO_PI), phi_fit, series, context, S)
    assert fit.measured and len(fit.windows) == 4
    for w in fit.windows:
        assert w.scale == pytest.approx(1.0, abs=0.01) and w.r2 > 0.999
        assert w.k_profile == pytest.approx(k_expected, rel=0.01)
        assert abs(math.degrees(np.angle(np.exp(1j * (w.phi0_profile - phi0_expected))))) < 1.0
        # fitted against the band-passed prediction (the same filter): K and phi0 agree
        assert w.k_fit / w.k_theory == pytest.approx(1.0, abs=0.01)
        assert abs(math.degrees(np.angle(np.exp(1j * (w.phi0_fit - w.phi0_theory))))) < 1.0
    assert owner.potential_accepts(fit, S) == (True, '')
    # the wrong sign (a body on the owner's maximum) is out of the band
    flipped = replace(fit, windows=tuple(replace(w, scale=-w.scale) for w in fit.windows))
    assert owner.potential_accepts(flipped, S) == (False, owner.REASON_POTENTIAL_SCALE)


def test_potential_route_applies_to_two_body_candidates_with_saved_planets():
    spec, series, _, _, mu, _ = _potential_record()
    context = SimpleNamespace(planet_mass={'Jupiter': mu})
    assert owner_potential.not_applicable(spec, series, context, S) == ''
    assert owner_potential.not_applicable(ResonanceSpec.from_key('1V+3S-7+0+0+3'), series, context, S) == 'three_body_owner'
    assert owner_potential.not_applicable(ResonanceSpec.from_key('1J-1+0+0'), series, context, S) == 'coorbital_owner'
    assert owner_potential.not_applicable(ResonanceSpec.from_key('2S-5+0+3'), series, context, S) == 'catalogue'
    assert owner_potential.not_applicable(spec, series, SimpleNamespace(planet_mass={}), S) == 'no_planet_mass'
    assert owner_potential.not_applicable(spec, series, context, replace(S, mmr_owner_potential='off')) == 'off'


def test_periodic_derivative_of_a_harmonic():
    grid = TWO_PI * np.arange(256) / 256
    values = 3.0 + 2.0 * np.cos(grid - 0.4) + 0.5 * np.sin(3 * grid)
    expected = -2.0 * np.sin(grid - 0.4) + 1.5 * np.cos(3 * grid)
    assert np.allclose(owner_potential.periodic_derivative(values), expected, atol=1e-12)
    assert owner_potential.first_harmonic(values) == pytest.approx(2.0)


def _fit(regimes):
    """An OwnerFit of 1J-2+0+1 (the record of `_potential_record` holds Jupiter: the route applies), psi share 0."""
    return owner.OwnerFit(
        ResonanceSpec.from_key('1J-2+0+1'), 0.01, 1.0, regimes, True, None, psi={r.regime: (0.0, 0.0, 0.0, 0.0) for r in regimes}
    )


def _window(scale, ratio=1.0):
    return owner_potential.PotentialWindow(0.0, 1.0, scale, 0.95, 1.0, 0.0, ratio, 0.0, 1.0, 0.0, 3.25, 0.12, 10.0, 720, 1e-6, 1e-3)


def test_the_potential_route_reads_every_usable_regime(monkeypatch):
    """FIX1 (Codex A #8): the best-R^2 regime (forced wave, R^2 0.95) is out of the a band (rate_outside_band) and was
    the only one the route looked at; the libration regime (R^2 0.85, phase incoherent) qualifies and its potential fits
    (scale 1.0, R^2 0.95): the owner is accepted by 'potential' on it, and only it counts for the intervals."""
    _, series, _, _, mu, _ = _potential_record()
    context = SimpleNamespace(planet_mass={'Jupiter': mu})
    forced = owner.RegimeFit(
        owner.REGIME_FORCED_POSITIVE, (0.0, 500.0), 0.95, (0.95, 0.95), 0.99, 1.0, 0.0, 0.0, 0.0, rate_admissible=False
    )
    libration = owner.RegimeFit(owner.REGIME_LIBRATION, (1000.0, 1500.0, 2000.0), 0.85, (0.85,) * 3, 0.3, 1.0, 0.0, 0.0, 0.0)
    judged = owner.judge(_fit([forced, libration]), SimpleNamespace(sigma_theory_au_yr=1.0), S)
    assert not judged.accepted and judged.reason == owner.REASON_RATE and judged.regime is forced
    measured = []

    def potential(item, *args):
        measured.append(item.regime.regime)
        return owner_potential.PotentialFit('', (_window(1.0),), 'Jupiter')

    monkeypatch.setattr(owner, 'potential_of', potential)
    owner.potential_pass([judged], None, series, context, S)
    assert measured == [owner.REGIME_LIBRATION]  # the forced regime fails the gates (its rate), not measured
    assert judged.accepted and judged.reason == 'accepted:potential' and judged.candidate.route == 'potential'
    assert judged.regime is libration and judged.passing == [libration]


def test_a_confirmation_that_rejects_every_owner_lets_the_potential_try_the_rest(monkeypatch):
    """FIX1 (Codex A #9): with MMR_OWNER_POTENTIAL_CONFIRM the owner accepted by R^2 (gain 0.95 over sigma's 0) has a
    measured force 5x the directly averaged one (outside [1/2, 2]): rejected (potential_unconfirmed); the other
    candidate, phase-incoherent, is then accepted by the potential. Off (default), the first owner stays."""
    _, series, _, _, mu, _ = _potential_record()
    context = SimpleNamespace(planet_mass={'Jupiter': mu})
    frame = SimpleNamespace(sigma_theory_au_yr=1.0)

    def candidates():
        strong = owner.RegimeFit(owner.REGIME_LIBRATION, (0.0, 500.0), 0.95, (0.95, 0.95), 0.99, 1.0, 0.0, 0.0, 0.0)
        weak = owner.RegimeFit(owner.REGIME_LIBRATION, (0.0, 500.0), 0.85, (0.85, 0.85), 0.3, 1.0, 0.0, 0.0, 0.0)
        first, second = owner.judge(_fit([strong]), frame, S), owner.judge(_fit([weak]), frame, S)
        assert first.accepted and first.reason == 'accepted:r2' and not second.accepted and second.reason == owner.REASON_COHERENCE
        return first, second

    def potential(item, *args):
        return owner_potential.PotentialFit('', (_window(1.0, ratio=5.0 if item.candidate.r2_median > 0.9 else 1.0),), 'Jupiter')

    monkeypatch.setattr(owner, 'potential_of', potential)
    first, second = candidates()
    owner.potential_pass([first, second], None, series, context, replace(S, mmr_owner_potential_confirm=True))
    assert not first.accepted and first.reason == owner.REASON_POTENTIAL_UNCONFIRMED and first.passing == []
    assert second.accepted and second.candidate.route == 'potential'
    first, second = candidates()
    owner.potential_pass([first, second], None, series, context, S)
    assert first.accepted and first.candidate.route == 'r2' and not second.accepted
    assert first.candidate.potential is not None and second.candidate.potential is None


# --------------------------------------------------------------------------------------------- identity text


def test_identity_text_of_385266():
    sigma, phi = ResonanceSpec.from_key('16N-27+0+11'), ResonanceSpec.from_key('3N-5+0+2')
    assert owner.identity_text(sigma, phi) == '(27/5)φ + (ϖ − λ_N)/5'
    assert owner.psi_definition(sigma, phi) == '5*sigma-27*phi'
    assert owner.psi_terms(sigma, phi) == {'varpi': 1, 'lambda:Neptune': -1}
    # GK3: 3M+5S-3+0+0-5 with 10E-17+0+7, Psi = 17 sigma - 3 phi = -106 varpi - 30 lambda_E + 51 lambda_M + 85 lambda_S
    gk3 = owner.psi_terms(ResonanceSpec.from_key('3M+5S-3+0+0-5'), ResonanceSpec.from_key('10E-17+0+7'))
    assert gk3 == {'varpi': -106, 'lambda:Earth': -30, 'lambda:Mars': 51, 'lambda:Saturn': 85}
    assert owner.identity_text(ResonanceSpec.from_key('3M+5S-3+0+0-5'), ResonanceSpec.from_key('10E-17+0+7')) == (
        '(3/17)φ + (−106ϖ − 30λ_E + 51λ_M + 85λ_S)/17'
    )


# --------------------------------------------------------------------------------------------- intervals, cover, F7


def _regime(kind, starts):
    return owner.RegimeFit(kind, tuple(starts), 0.9, (0.9,) * len(starts), 0.9, 1.0, 0.0, 0.0, 0.0)


def test_owner_intervals_slips_and_cover_by_hand():
    librations = _regime(owner.REGIME_LIBRATION, [0, 50, 100, 400, 450])
    forced = _regime(owner.REGIME_FORCED_POSITIVE, [250])
    intervals = owner.owner_intervals([librations, forced], 100.0)
    # libration 0-200 and 400-550, forced 250-350, slips 200-250 and 350-400 between the librations
    assert intervals == [
        (0.0, 200.0, 'libration'),
        (200.0, 250.0, 'slip'),
        (250.0, 350.0, 'forced_wave'),
        (350.0, 400.0, 'slip'),
        (400.0, 550.0, 'libration'),
    ]
    assert owner.connected_blocks(intervals) == [(0.0, 550.0)]
    assert owner.covered_fraction(intervals, (100.0, 300.0)) == pytest.approx(1.0)
    assert owner.covered_fraction([(0.0, 50.0, 'libration'), (80.0, 100.0, 'libration')], (0.0, 100.0)) == pytest.approx(0.7)
    assert owner.covered_fraction(intervals, (600.0, 700.0)) == 0.0
    assert intervals_text(intervals[:2], 0.25) == '0.0-200.0:libration;200.0-250.0:slip'  # the step's order: 1 decimal
    assert intervals_text([(11585.33, 14639.07, 'libration')], 5.0) == '11585-14640:libration'


def _owner_result(cover):
    accepted = OwnerCandidate(
        spec=ResonanceSpec.from_key('27E-47+0+20'),
        rate_rad_yr=0.0,
        r2_median=0.96,
        r2_windows=(0.96,) * 10,
        phi0_coherence=0.99,
        psi_share=0.0,
        force_au_yr=1.0,
        force_theory_au_yr=1.0,
        strength_ratio=0.005,
        rank=1,
        accepted=True,
        reason='accepted:r2',
        route='r2',
    )
    return OwnerResult(1, [accepted], accepted, CheckValue.FAIL, 0.0, NAN, NAN, NAN, NAN, NAN, 194.0, 'owner_libration', cover=cover)


def test_an_owner_outside_the_episode_does_not_take_it():
    ledger = Ledger.from_settings(S)
    kept, flags = decide.owner_in_episode(_owner_result(0.0), S, ledger)  # 2001 PG14 step 0.25 yr: 27E-47 at 11.6-19 kyr
    assert kept is None and flags == [str(Flag.OWNER_OUTSIDE_EPISODE)]
    entry = [e for e in ledger if e.name.startswith('2 owner cover')][0]
    assert entry.result is False and entry.threshold == S.mmr_owner_cover_min and entry.column == 'own_cover'
    for cover in (S.mmr_owner_cover_min, 0.93):
        kept, flags = decide.owner_in_episode(_owner_result(cover), S, Ledger.from_settings(S))
        assert kept is not None and flags == []
    kept, flags = decide.owner_in_episode(_owner_result(NAN), S, Ledger.from_settings(S))  # not measured: kept
    assert kept is not None and flags == []


def test_on_a_circulation_the_owner_cover_is_no_thin_margin():
    """237743 5J-2S-2 (W20 round C): cover 0.538 against the convention 0.5 is within the thin band, a U margin when
    the owner may take sigma's libration; on a circulation the owner only names the origin of a non_resonant verdict,
    so the same comparison is recorded without its setting: no convention, no U. The decision (kept or not) is the same."""
    cover = 0.538
    assert abs(cover / S.mmr_owner_cover_min - 1) < S.confidence_thin_band
    decisive = Ledger.from_settings(S)
    kept, _ = decide.owner_in_episode(_owner_result(cover), S, decisive)
    assert kept is not None and decisive.thin_margins() == ['U:2 owner cover of the decisive episode']
    naming = Ledger.from_settings(S)
    kept, _ = decide.owner_in_episode(_owner_result(cover), S, naming, decisive=False)
    entry = [e for e in naming if e.name.startswith('2 owner cover')][0]
    assert kept is not None and naming.thin_margins() == [] and entry.result is True and entry.setting == ''


def test_second_capture_caps_unless_excluded():
    result = _owner_result(1.0)
    ledger = Ledger.from_settings(S)
    assert decide.second_capture_caps(result, S, ledger) == (['second_capture_unresolved'], ['second_capture_unresolved'])
    result.capture = owner.SecondCapture(3.0, math.radians(243.0), 0.26, 0.03, owner.CAPTURE_EXCLUDED)
    assert decide.second_capture_caps(result, S, Ledger.from_settings(S)) == ([], [])
