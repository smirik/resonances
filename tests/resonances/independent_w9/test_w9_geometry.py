"""Independent tests of W9 (geometry: the slow edge half-cycle; strength: the well rule; decide: G1l, rule C, L4
without the phi0 spread; planets: the great-inequality ripple and the catalogue longitudes).

Expected values are derived here, not read from the code:

- geometry: the band test of docs/physics/geometry.md ("Runs and episodes") written out (the band of a full cycle TP_i..TP_i+2 is
  m ± A with m, A the means of its two half-cycle midpoints and half-amplitudes, i.e. [middle TP, mean of the outer
  TPs]); for a uniform drift b per half-cycle its margin is 2A - (N - 3) b. The runs below are built so that each
  cut (none, left, right, both) passes or fails by a margin of >= 0.1 rad under that rule;
- strength: <D> = 1/|r - r_P| - r.r_P/r_P^3 averaged by brute force (own Kepler solver) along the line
  M = (sigma - k_P lambda_P) / k of an exterior 1:2 with Neptune: two asymmetric wells at about ±76 deg, the global
  maximum at 0 (conjunction at perihelion), an inner maximum at 180 deg (Gallardo 2006, sect. 3.1);
- decide: a pendulum record of 3J-2 (sigma = 3 lambda_J - 2 lambda - varpi, a from n by Kepler) that circulates at
  2 pi/100 yr and is captured at 9 kyr; the jump of a is computed from n; a force law a_dot = R sin(sigma - phi0(t))
  with phi0 drifting by 90 deg; rule C from the docstring of `member_status`;
- planets: r_GI = |k_J dl_J - k_S dl_S| G(P_GI) with G = 1/(1 + (tan(pi dt/P)/tan(pi dt/P_c))^(2N)) by hand;
  D = sum |k_j| min(SPREAD T, 2 SLOW_MAX + OFFSET T) by hand from the const tables; 0.2181 = max |d sinc(D/2)/dD|.
"""

import dataclasses
import math

import numpy as np
import pytest

from resonances.data import const
from resonances.resonance.analysis.geometry import (
    Episode,
    EpisodeShape,
    RejectReason,
    TurningPoints,
    drift_fraction,
    find_episodes,
    short_runs,
)
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import averaged_potential as av
from resonances.resonance.analysis.mmr import context, owner
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.coorbital import Orbit
from resonances.resonance.analysis.mmr.decide import AngleDecision, member_status, members_locked
from resonances.resonance.analysis.mmr.models import (
    GreatInequality,
    MMRSeries,
    PlanetSeries,
    Regime,
    ResonanceSpec,
    SubresonanceCandidate,
    SubresonanceResult,
)
from resonances.resonance.analysis.mmr.preprocess import predicted_ripple
from resonances.resonance.analysis.mmr.response import a_response_check, force_windows, phase_spread
from resonances.resonance.analysis.mmr.strength import observed_centre, observed_swing, quadrature
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import (
    AngleStatus,
    AngleSubtype,
    CheckValue,
    LibrationOrigin,
    Reason,
    ResonanceType,
    Status,
    Verdict,
)

TWO_PI = 2.0 * np.pi
S = AnalysisSettings()
H = 500.0  # the regular half-period of the synthetic runs [yr]


# ================================================================================================ geometry


def reference_bands_overlap(values):
    """The band test of docs §3 on TP values: every full cycle's band m ± A shares a point (strict)."""
    v = np.asarray(values, float)
    lows, highs = [], []
    for i in range(len(v) - 2):
        m = ((v[i] + v[i + 1]) / 2 + (v[i + 1] + v[i + 2]) / 2) / 2
        a = (abs(v[i] - v[i + 1]) / 2 + abs(v[i + 1] - v[i + 2]) / 2) / 2
        lows.append(m - a)
        highs.append(m + a)
    return max(lows) < min(highs), min(highs) - max(lows)


def turning_points(halves, values, first_kind, t0=1000.0):
    times = np.concatenate([[t0], t0 + np.cumsum(halves)])
    values = np.asarray(values, float)
    assert len(times) == len(values)
    kinds = np.array([first_kind * (-1) ** k for k in range(len(values))])
    for k in range(1, len(values)):  # alternation: each maximum above its neighbouring minima
        assert (values[k] - values[k - 1]) * kinds[k] > 0
    return TurningPoints(np.round(times).astype(int), times, values, kinds, 0.01)


def episodes_of(tp, tail=1000.0):
    t = np.arange(0.0, tp.times_yr[-1] + tail + 1.0)
    window = (float(t[0]), float(t[-1]))
    return find_episodes(t, tp, window, S), window


def drift_values(n, b, first_kind, amp=1.0):
    """v_k = b k + first_kind (-1)^k A: a uniform drift of b per half-cycle."""
    return [b * k + first_kind * amp * (-1) ** k for k in range(n)]


def gq107_like():
    """13 TPs from a minimum, a drift b = 0.15 rad per half-cycle (12 TPs: margin 2 - 9b = 0.65), the last minimum
    raised by 1.5 rad and reached after 2.4 half-periods (13 TPs: margin 2 - 10b - 1.5/2 = -0.25). Half ratio 2.4 > 2,
    full period (500 + 1200) / 1000 = 1.7 < 2: `_split_regular` cannot see it (GQ107: 2.07 and 1.74)."""
    values = drift_values(13, 0.15, -1)
    values[12] += 1.5
    halves = [H] * 11 + [2.4 * H]
    return halves, values


def test_reference_rule_on_the_designs():
    """The hand numbers behind the designs below (the band rule itself, not the code)."""
    halves, values = gq107_like()
    assert not reference_bands_overlap(values)[0]
    ok, margin = reference_bands_overlap(values[:12])
    assert ok and margin == pytest.approx(2 - 9 * 0.15)
    for n, b in ((13, 0.3), (12, 0.3)):
        assert reference_bands_overlap(drift_values(n, b, -1))[1] == pytest.approx(2 - (n - 3) * b)
    # the half-period ratio and the full-period ratio of the slow edge
    assert halves[-1] / np.median(halves) == pytest.approx(2.4)
    full = np.array(halves[:-1]) + np.array(halves[1:])
    assert full[-1] / np.median(full) == pytest.approx(1.7)


def test_slow_drifting_last_half_cycle_is_trimmed_and_the_core_accepted():
    halves, values = gq107_like()
    tp = turning_points(halves, values, -1)
    runs, window = episodes_of(tp)
    assert [(r.episode_id, r.n_tp, r.accepted) for r in runs] == [(0, 12, True), (1, 1, False)]
    core, edge = runs
    assert core.shape == EpisodeShape.STATIONARY
    assert core.tp_times_yr[-1] == pytest.approx(tp.times_yr[11])
    assert edge.reject_reason == RejectReason.EDGE
    assert edge.tp_times_yr[0] == pytest.approx(tp.times_yr[12])
    # the trimmed half-cycle counts neither as a near-episode nor as drift
    assert short_runs(runs) == []
    assert drift_fraction(runs, window, S) == 0.0


def test_same_drifting_edge_at_regular_pace_is_not_trimmed():
    """The rule needs a slow half-cycle: the same values at equal half-periods stay one centre_moves run."""
    _, values = gq107_like()
    runs, _ = episodes_of(turning_points([H] * 12, values, -1))
    assert [(r.n_tp, r.accepted, r.reject_reason) for r in runs] == [(13, False, RejectReason.CENTRE_MOVES)]


@pytest.mark.parametrize('halves', [[H] * 12, [H] * 11 + [2.4 * H]], ids=['equal_halves', 'slow_last_half'])
def test_uniform_drift_is_not_accepted_on_a_shorter_piece(halves):
    """b = 0.3: 13 TPs margin -1.0, 12 TPs -0.7, but 5 TPs would pass (+1.4): no cut beyond one half-cycle."""
    values = drift_values(13, 0.3, -1)
    assert reference_bands_overlap(values[:5])[0]  # what an iterated trim would accept
    runs, _ = episodes_of(turning_points(halves, values, -1))
    assert [(r.n_tp, r.accepted, r.reject_reason) for r in runs] == [(13, False, RejectReason.CENTRE_MOVES)]
    assert not any(r.reject_reason == RejectReason.EDGE for r in runs)


def test_two_slow_half_cycles_at_the_start_only_one_is_cut():
    """b = 0.235: 13 TPs margin -0.35, 12 TPs -0.115 (one cut), 11 TPs +0.12 (two cuts): nothing changes."""
    values = drift_values(13, 0.235, -1)
    assert not reference_bands_overlap(values[1:])[0]
    assert reference_bands_overlap(values[2:])[0]
    runs, _ = episodes_of(turning_points([2.4 * H, 2.4 * H] + [H] * 10, values, -1))
    assert [(r.n_tp, r.accepted, r.reject_reason) for r in runs] == [(13, False, RejectReason.CENTRE_MOVES)]


def test_accepted_run_with_a_slow_edge_is_untouched():
    values = drift_values(13, 0.0, -1)
    values[12] += 0.8
    assert reference_bands_overlap(values)[0]
    runs, _ = episodes_of(turning_points([H] * 11 + [2.4 * H], values, -1))
    assert [(r.episode_id, r.n_tp, r.accepted) for r in runs] == [(0, 13, True)]


def both_edges(b, shift):
    """14 TPs from a maximum, drift b; the first maximum lowered and the last minimum raised by `shift`."""
    values = drift_values(14, b, +1)
    values[0] -= shift
    values[13] += shift
    return values


def test_both_slow_edges_removed_when_each_alone_is_not_enough():
    """b = 0.15, shift 1.6: left cut alone margin 2 - 10b - 0.8 = -0.3, right alone the same, both 2 - 9b = +0.65."""
    values = both_edges(0.15, 1.6)
    assert not reference_bands_overlap(values[1:])[0]
    assert not reference_bands_overlap(values[:-1])[0]
    assert reference_bands_overlap(values[1:-1])[0]
    tp = turning_points([2.4 * H] + [H] * 11 + [2.4 * H], values, +1)
    runs, _ = episodes_of(tp)
    assert [(r.episode_id, r.n_tp, r.accepted, r.reject_reason) for r in runs] == [
        (0, 1, False, RejectReason.EDGE),
        (1, 12, True, RejectReason.NONE),
        (2, 1, False, RejectReason.EDGE),
    ]


@pytest.mark.parametrize(
    'left, right, cut',
    [(1300.0, 1100.0, 'right'), (1100.0, 1300.0, 'left'), (1200.0, 1200.0, 'right')],
    ids=['longer_core_keeps_the_slower_right', 'longer_core_keeps_the_slower_left', 'tie_takes_the_earlier_core'],
)
def test_selection_most_tps_then_longer_then_earlier(left, right, cut):
    """b = 0.05, shift 1.7: whole run margin -0.25, either single cut passes (13 TPs), both cuts pass (12 TPs).
    Most TPs: a single cut. Longer core: remove the shorter slow half-cycle. Equal: the earlier core (cut right)."""
    values = both_edges(0.05, 1.7)
    assert not reference_bands_overlap(values)[0]
    assert reference_bands_overlap(values[1:])[0] and reference_bands_overlap(values[:-1])[0]
    tp = turning_points([left] + [H] * 11 + [right], values, +1)
    runs, _ = episodes_of(tp)
    if cut == 'right':
        expected = [(0, 13, True, RejectReason.NONE), (1, 1, False, RejectReason.EDGE)]
    else:
        expected = [(0, 1, False, RejectReason.EDGE), (1, 13, True, RejectReason.NONE)]
    assert [(r.episode_id, r.n_tp, r.accepted, r.reject_reason) for r in runs] == expected


def test_ids_stay_consecutive_across_runs():
    """The trimmed run, a 5000-yr gap (> 3 x the median half-period: a new run), then a stationary run of 5 TPs."""
    halves, values = gq107_like()
    second = [5.0 + (-1) ** k for k in range(5)]  # after the last minimum (2.3 rad) a maximum first, centre 5 rad
    tp = turning_points(halves + [5000.0] + [H] * 4, values + second, -1)
    runs, _ = episodes_of(tp)
    assert [(r.episode_id, r.n_tp, r.accepted, r.reject_reason) for r in runs] == [
        (0, 12, True, RejectReason.NONE),
        (1, 1, False, RejectReason.EDGE),
        (2, 5, True, RejectReason.NONE),
    ]


# ================================================================================================ strength

A_NEPTUNE = 30.07
K_BODY, K_PLANET = -2, 1  # sigma = lambda_N - 2 lambda (+ varpi, zero here): exterior 1:2
A_BODY = A_NEPTUNE * 2 ** (2 / 3)
E_BODY = 0.25


def kepler_position(a, e, mean_anomaly):
    m = np.mod(mean_anomaly, TWO_PI)
    ecc = m.copy()
    for _ in range(60):
        ecc = ecc - (ecc - e * np.sin(ecc) - m) / (1 - e * np.cos(ecc))
    return np.stack([a * (np.cos(ecc) - e), a * np.sqrt(1 - e * e) * np.sin(ecc), 0 * ecc])


def brute_average(sigma, nodes_per_turn=8192):
    """<D>(sigma) over |k| turns of lambda_P, M = (sigma - k_P lambda_P) / k; planar, varpi = 0, circular planet."""
    lam = np.linspace(0, TWO_PI * abs(K_BODY), nodes_per_turn * abs(K_BODY), endpoint=False)
    planet = np.stack([A_NEPTUNE * np.cos(lam), A_NEPTUNE * np.sin(lam), 0 * lam])
    out = []
    for s in np.atleast_1d(sigma):
        body = kepler_position(A_BODY, E_BODY, (s - K_PLANET * lam) / K_BODY)
        d = np.linalg.norm(body - planet, axis=0)
        out.append(np.mean(1 / d - (body * planet).sum(0) / A_NEPTUNE**3))
    return np.array(out)


@pytest.fixture(scope='module')
def neptune_1_2():
    body = av.Ellipse(A_BODY, Orbit(E_BODY, 0.0, 0.0, 0.0))
    planet = av.Ellipse(A_NEPTUNE, Orbit(0.0, 0.0, 0.0, 0.0))
    sigma = np.radians(np.arange(0.0, 360.0, 0.25))
    values = brute_average(sigma, 4096)
    return body, planet, sigma, values


def test_averaged_profile_matches_brute_force(neptune_1_2):
    body, planet, _, _ = neptune_1_2
    sigma, profile = av.averaged_profile(body, planet, K_BODY, K_PLANET, 0.0, 256)
    brute = brute_average(sigma)
    assert np.max(np.abs(profile - brute)) < 1e-6 * np.ptp(brute)


def test_brute_force_potential_shape(neptune_1_2):
    """Two asymmetric wells, the global top at the conjunction at perihelion (0), an inner top at 180 deg."""
    _, _, sigma, values = neptune_1_2
    deg = np.degrees(sigma)
    minima = deg[(values < np.roll(values, 1)) & (values < np.roll(values, -1))]
    maxima = deg[(values > np.roll(values, 1)) & (values > np.roll(values, -1))]
    assert len(minima) == 2 and abs(minima[0] - (360 - minima[1])) < 0.5 and 60 < minima[0] < 90
    assert sorted(maxima) == [0.0, 180.0]
    assert values[0] > values[np.argmin(np.abs(deg - 180))]


@pytest.mark.parametrize('observed_deg, well', [(60.0, 'east'), (10.0, 'east'), (300.0, 'west'), (190.0, 'west')])
def test_centre_curvature_finds_the_basin_and_the_global_top(neptune_1_2, observed_deg, well):
    body, planet, sigma, values = neptune_1_2
    deg = np.degrees(sigma)
    half = deg < 180 if well == 'east' else deg > 180
    expected_centre = deg[half][np.argmin(values[half])]
    result = av.centre_curvature(body, planet, K_BODY, K_PLANET, 0.0, np.radians(observed_deg), quadrature(S))
    assert result.reason == ''
    assert np.degrees(av.circular_distance(result.centre_rad, np.radians(expected_centre))) < 0.6
    assert np.degrees(av.circular_distance(result.global_max_rad, 0.0)) < 360.0 / S.mmr_strength_sigma_grid
    # the curvature by central differences of the brute force about its own minimum (h = 1 deg)
    fine = np.radians(np.arange(expected_centre - 1.0, expected_centre + 1.0, 0.02))
    centre = fine[np.argmin(brute_average(fine, 4096))]
    h = np.radians(1.0)
    f = brute_average(np.array([centre - h, centre, centre + h]), 4096)
    curvature = (f[0] - 2 * f[1] + f[2]) / h**2
    assert result.curvature == pytest.approx(curvature, rel=0.02)


SWING_CASES = {
    # (lo, hi) of the observed half-cycle [deg]; centre 76 deg (east well), global top 0 deg
    'inside_the_east_well': ((40.0, 110.0), True),
    'same_swing_one_turn_up': ((400.0, 470.0), True),
    'horseshoe_over_the_inner_top_only': ((50.0, 290.0), True),
    'over_the_global_top': ((-40.0, 100.0), False),
    'over_the_global_top_one_turn_up': ((320.0, 460.0), False),
    'misses_the_centre': ((100.0, 170.0), False),
    'top_exactly_at_the_turn': ((0.0, 120.0), True),
}


@pytest.mark.parametrize('case', SWING_CASES, ids=list(SWING_CASES))
def test_swing_in_well(neptune_1_2, case):
    _, _, sigma, values = neptune_1_2
    deg = np.degrees(sigma)
    east = deg < 180
    centre = np.radians(deg[east][np.argmin(values[east])])
    top = sigma[np.argmax(values)]
    (lo, hi), expected = SWING_CASES[case]
    assert av.swing_in_well(centre, top, np.radians(lo), np.radians(hi)) is expected


def test_swing_in_well_without_a_centre_or_top():
    assert av.swing_in_well(float('nan'), 0.0, 0.0, 1.0) is False
    assert av.swing_in_well(0.5, float('nan'), 0.0, 1.0) is False


def episode_from(times, values):
    values = np.asarray(values, float)
    return Episode(
        episode_id=0,
        tp_times_yr=np.asarray(times, float),
        tp_values_rad=values,
        accepted=True,
        shape=EpisodeShape.STATIONARY,
        reject_reason=RejectReason.NONE,
        period_med_yr=200.0,
        period_min_yr=200.0,
        period_max_yr=200.0,
        half_amp_min_rad=0.5,
        half_amp_med_rad=0.5,
        half_amp_max_rad=0.5,
        centre_mod_2pi_rad=0.0,
        centre_shift_rad=0.0,
        centre_shift_over_amp=0.0,
        bands_overlap=True,
        centre_returns=0,
        coverage_start_gap_p=0.0,
        coverage_end_gap_p=0.0,
    )


def test_observed_swing_is_the_nearest_half_cycle_sorted():
    ep = episode_from([0.0, 100.0, 200.0, 300.0], [7.0, 5.5, 7.2, 5.6])  # unwrapped values beyond 2 pi
    assert observed_swing(ep, 150.0) == pytest.approx((5.5, 7.2))
    assert observed_swing(ep, 250.0) == pytest.approx((5.6, 7.2))
    assert observed_swing(ep, 40.0) == pytest.approx((5.5, 7.0))
    assert observed_centre(ep, 150.0) == pytest.approx(np.mod((5.5 + 7.2) / 2, TWO_PI))
    lo, hi = observed_swing(episode_from([0.0], [1.0]), 0.0)
    assert math.isnan(lo) and math.isnan(hi)


# ================================================================================================ decide

N_J = TWO_PI / 11.862
VARPI_RATE = TWO_PI / 2.5e5


def hilda_record(sigma, t):
    """A 3J-2+0-1 record whose sigma is `sigma`: lambda = (3 lambda_J - varpi - sigma) / 2, a = (2 pi / n)^(2/3)."""
    lam_j = 0.3 + N_J * t
    varpi = 1.0 + VARPI_RATE * t
    lam = (3 * lam_j - varpi - sigma) / 2
    a = (TWO_PI / np.gradient(lam, t)) ** (2 / 3)
    node = 2.0 - TWO_PI / 3e5 * t
    ones = np.ones_like(t)
    jupiter = PlanetSeries('Jupiter', t, 5.2026 * ones, 0.048 * ones, 0.0228 * ones, 1.75 * ones, 0.25 * ones, np.mod(lam_j, TWO_PI))
    return MMRSeries(
        t,
        a,
        0.15 * ones,
        0.1 * ones,
        np.mod(node, TWO_PI),
        np.mod(varpi - node, TWO_PI),
        np.mod(varpi, TWO_PI),
        np.mod(sigma, TWO_PI),
        ResonanceSpec.from_key('3J-2+0-1'),
        Regime(t[1] - t[0], t[-1] - t[0], len(t), 'ias15'),
        {'Jupiter': jupiter},
        np.mod(lam, TWO_PI),
        np.mod(lam - varpi, TWO_PI),
        name='capture',
    )


T_CAPTURE, P_CIRC, P_LIB, AMP_DEG = 9000.0, 100.0, 250.0, 40.0


@pytest.fixture(scope='module')
def capture_analysis():
    t = np.arange(0.0, 20000.5, 1.0)
    circulation = TWO_PI * np.minimum(t, T_CAPTURE) / P_CIRC
    libration = np.radians(AMP_DEG) * np.sin(TWO_PI * (t - T_CAPTURE) / P_LIB)
    sigma = np.where(t < T_CAPTURE, circulation, circulation + libration)
    return analyse_mmr(hilda_record(sigma, t), S)


def test_capture_after_a_jump_is_transient_with_final_regime_libration(capture_analysis):
    verdict = capture_analysis.verdict
    numbers = capture_analysis.numbers_row()
    # the jump of a: n before = (3 n_J - varpi_dot - 2 pi / P_circ) / 2, after = (3 n_J - varpi_dot) / 2
    n_after = (3 * N_J - VARPI_RATE) / 2
    n_before = n_after - TWO_PI / P_CIRC / 2
    jump = (TWO_PI / n_after) ** (2 / 3) - (TWO_PI / n_before) ** (2 / 3)
    assert jump == pytest.approx(-0.108, abs=0.002)
    sizes = [float(x) for x in str(numbers['jump_sizes_au']).split(';') if x != '']
    # the locator reports the largest increment a_f(t + tau/2) - a_f(t - tau/2), tau = 2 P_c = 150 yr: it holds the
    # step and up to one half-swing of the libration's a-wave, delta a = (2/3) a (A w / 2) / n (n = (3 n_J - varpi_dot
    # - sigma_dot) / 2, A = 40 deg, w = 2 pi / 250 yr): 0.029 au, with the 5 % tolerance the test had before
    a_after = (TWO_PI / n_after) ** (2 / 3)
    swing = (2 / 3) * a_after * (np.radians(AMP_DEG) * TWO_PI / P_LIB / 2) / n_after
    assert swing == pytest.approx(0.0292, abs=5e-4)
    assert len(sizes) == 1 and jump - 1.05 * swing <= sizes[0] <= jump + 1.05 * swing
    assert abs(float(str(numbers['jump_times_yr'])) - T_CAPTURE) < 2 * P_LIB
    # the rule and the verdict
    assert any(e.name.startswith('G1l') for e in verdict.decision_path)
    assert numbers['geo_n_episodes_before_jump'] == 0
    assert verdict.angle_status == AngleStatus.EPISODIC
    assert verdict.status == Status.TRANSIENT
    assert verdict.final_regime == AngleStatus.LIBRATION
    assert T_CAPTURE <= verdict.final_regime_start_yr <= T_CAPTURE + P_LIB


@pytest.mark.parametrize('drift_deg', [0.0, 90.0])
def test_phase_shifted_clean_force_law_passes_l4(drift_deg):
    """a_dot = R sin(sigma - pi - phi0(t)) + C, phi0 drifting linearly by `drift_deg` over 20 kyr: R^2 ~ 1 in every
    window. At 90 deg the largest |phi0 - mean| is ~ 45 deg > MMR_FORCE_PHASE_TOL_DEG, which no longer decides."""
    t = np.arange(0.0, 20000.0, 2.0)
    period = 500.0
    sigma = np.pi + np.radians(50) * np.sin(TWO_PI * t / period)
    phi0 = np.radians(drift_deg) * t / t[-1]
    adot = 1e-4 * np.sin(sigma - np.pi - phi0) + 2e-6
    windows = force_windows(t, sigma, adot, S.mmr_force_window_cycles * period, S.mmr_force_window_step)
    assert len(windows) >= S.mmr_force_min_windows
    assert min(w.r2 for w in windows) > S.mmr_force_r2_full
    if drift_deg:
        assert np.degrees(phase_spread(windows)[2]) > S.mmr_force_phase_tol_deg
    check, thin, law = a_response_check(windows, None, np.radians(50), S)
    assert (check, thin, law) == (CheckValue.PASS, False, 'single')


def test_force_law_without_sigma_dependence_fails_l4():
    rng = np.random.default_rng(1)
    t = np.arange(0.0, 20000.0, 2.0)
    sigma = np.pi + np.radians(50) * np.sin(TWO_PI * t / 500.0)
    adot = 1e-4 * rng.standard_normal(len(t))
    windows = force_windows(t, sigma, adot, S.mmr_force_window_cycles * 500.0, S.mmr_force_window_step)
    assert a_response_check(windows, None, np.radians(50), S)[0] == CheckValue.FAIL


CATALOGUE_KEY, MEMBER_KEY = '13M-24+0+11', '13M-24+2+9'


def subresonances_with_member():
    member = SubresonanceCandidate(
        ResonanceSpec.from_key(MEMBER_KEY), 2, False, 0.1, 0.0, 0.1, 0.0, 0.99, 1.0, 0.1, 0.9, 1.0, 1, True, 'accepted'
    )
    return SubresonanceResult((), (), [member], member, CheckValue.PASS, False, False, False, True, 0.99, 0.1, 0.2, 2, 0.0, 0.0, 0.0)


def member_verdict(status, angle_status):
    return Verdict(ResonanceType.MMR, status, angle_status)


ANGLES = {
    'circulation': AngleDecision(AngleStatus.CIRCULATION, AngleSubtype.NONE, 'G4'),
    'episodic': AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, 'G2'),
    'libration': AngleDecision(AngleStatus.LIBRATION, AngleSubtype.STATIONARY, 'G2'),
}


@pytest.mark.parametrize(
    'angle, member, expected',
    [
        ('circulation', (Status.RESONANT, AngleStatus.LIBRATION), Status.RESONANT),
        ('circulation', (Status.TRANSIENT, AngleStatus.EPISODIC), Status.TRANSIENT),
        ('episodic', (Status.RESONANT, AngleStatus.LIBRATION), Status.RESONANT),
        ('circulation', (Status.INVALID, AngleStatus.UNDEFINED), Status.UNCERTAIN),
    ],
)
def test_catalogue_row_takes_the_librating_members_status(angle, member, expected):
    decision = member_status(ANGLES[angle], subresonances_with_member(), member_verdict(*member), Ledger(S.confidence_thin_band))
    assert decision is not None
    assert decision.status == expected
    assert decision.reason == Reason.SUBRESONANCE_LIBRATES
    assert decision.origin == LibrationOrigin.SUBRESONANCE
    assert decision.owner == MEMBER_KEY


def test_both_librating_keep_their_own_verdicts():
    ledger = Ledger(S.confidence_thin_band)
    member = member_verdict(Status.RESONANT, AngleStatus.LIBRATION)
    assert member_status(ANGLES['libration'], subresonances_with_member(), member, ledger) is None
    assert members_locked(ANGLES['libration'], member) == 1
    assert ledger.entries  # the decision is written down


def test_no_member_row_leaves_rule_c_silent():
    for angle in ANGLES.values():
        assert member_status(angle, subresonances_with_member(), None, Ledger(S.confidence_thin_band)) is None
    assert math.isnan(members_locked(ANGLES['libration'], None))


# ================================================================================================ planets

GI = GreatInequality(937.0, np.radians(0.34), np.radians(0.83), False)


def analog_gain(period, cutoff, order):
    return 1 / (1 + (cutoff / period) ** (2 * order))


def digital_gain(period, cutoff, order, step):
    return 1 / (1 + (math.tan(math.pi * step / period) / math.tan(math.pi * step / cutoff)) ** (2 * order))


RIPPLE_CASES = {
    # key: hand half-amplitude before the filter [deg]
    '5J+6S-1+0+0-10': abs(5 * 0.34 - 6 * 0.83),  # same signs: the antiphase waves cancel, 3.28 not 6.68
    '2J+1S-1+0+0-2': abs(2 * 0.34 - 1 * 0.83),  # 0.15
    '5J-2S-2+0+0-1': abs(5 * 0.34 + 2 * 0.83),  # opposite signs add: 3.36
    '3J-2+0-1': 3 * 0.34,  # Jupiter alone
}


@pytest.mark.parametrize('key', RIPPLE_CASES)
@pytest.mark.parametrize('cutoff', [937.0, 468.5])
def test_predicted_ripple_by_hand(key, cutoff):
    spec = ResonanceSpec.from_key(key)
    order = S.filter_order
    expected = np.radians(RIPPLE_CASES[key]) * analog_gain(937.0, cutoff, order)
    assert predicted_ripple(spec, GI, cutoff, order) == pytest.approx(expected, rel=1e-9)
    step = 5.0
    expected_digital = np.radians(RIPPLE_CASES[key]) * digital_gain(937.0, cutoff, order, step)
    assert predicted_ripple(spec, GI, cutoff, order, step) == pytest.approx(expected_digital, rel=1e-6)


def test_ripple_numbers_quoted_in_the_docs():
    assert RIPPLE_CASES['5J+6S-1+0+0-10'] == pytest.approx(3.28)
    assert analog_gain(937.0, 937.0, 2) == pytest.approx(0.5)
    assert analog_gain(937.0, 468.5, 2) == pytest.approx(16 / 17)


def hand_phase_error(coefficients, span):
    total = 0.0
    for name, k in coefficients.items():
        spread = const.PLANET_N_SPREAD_DEG_PER_GAUSS_YR[name] * span
        offset = 2 * const.PLANET_LAMBDA_SLOW_MAX_DEG[name] + const.PLANET_N_OFFSET_DEG_PER_GAUSS_YR[name] * span
        total += abs(k) * min(spread, offset)
    return np.radians(total)


@pytest.mark.parametrize(
    'catalogued, span, expected_deg',
    [
        (('Jupiter', 'Saturn'), 20000.0, 5 * 3.0 + 2 * 7.4),  # SPREAD T branch for both
        (('Jupiter', 'Saturn'), 200000.0, 5 * 4.64 + 2 * 11.08),  # 2 SLOW_MAX + OFFSET T branch for both
        (('Saturn',), 20000.0, 2 * 7.4),  # Jupiter saved: no error from it
        ((), 20000.0, 0.0),
    ],
)
def test_catalogue_phase_error_bound(catalogued, span, expected_deg):
    spec = ResonanceSpec.from_key('5J-2S-2+0+0-1')
    coefficients = {name: k for name, k in zip(spec.planets, spec.planet_coefficients) if name in catalogued}
    assert hand_phase_error(coefficients, span) == pytest.approx(np.radians(expected_deg), rel=1e-9)
    assert owner.catalogue_phase_error(spec, catalogued, span) == pytest.approx(np.radians(expected_deg), rel=1e-9)


def test_sinc_half_slope_constant():
    """The coherence of a phase drifting by D over the span is |sinc(D/2)| = |sin(D/2)/(D/2)|; its largest slope in D
    is max_u |(u cos u - sin u)/u^2| / 2 (u = D/2)."""
    u = np.linspace(1e-4, 20.0, 400001)
    slope = np.max(np.abs((u * np.cos(u) - np.sin(u)) / u**2)) / 2
    assert owner.SINC_HALF_SLOPE_MAX == pytest.approx(slope, abs=1e-4)
    # and it bounds the change of the coherence of a linear drift: |C(D + d) - C(D)| <= slope x d
    drift = np.linspace(0.0, 12.0, 2001)
    coherence = np.array([abs(np.mean(np.exp(1j * d * np.linspace(-0.5, 0.5, 4001)))) for d in drift])
    assert np.max(np.abs(np.diff(coherence)) / np.diff(drift)) <= owner.SINC_HALF_SLOPE_MAX + 1e-3


def test_catalogue_longitudes_carry_lambda_with_the_table_rate():
    out = context.catalogue_longitudes({'Mercury': 1.0, 'Jupiter': 1.0, 'Earth': 6.0}, 100.0, 0.0)
    assert set(out) == {'Jupiter', 'Earth'}  # Mercury is never catalogued
    jupiter = np.mod(1.0 - np.radians(const.PLANET_N_DEG_PER_GAUSS_YR['Jupiter']) * 100.0, TWO_PI)
    earth = np.mod(6.0 - np.radians(const.PLANET_N_DEG_PER_GAUSS_YR['Earth']) * 100.0, TWO_PI)
    assert out['Jupiter'] == pytest.approx(jupiter, abs=1e-12)
    assert out['Earth'] == pytest.approx(earth, abs=1e-12)
    assert 'Mercury' not in context.CATALOGUE_PLANETS


def test_catalogue_line_of_an_unsaved_planet():
    t = np.arange(100.0, 1100.0, 10.0)
    sigma = np.pi + 0.5 * np.sin(TWO_PI * t / 250.0)
    series = dataclasses.replace(hilda_record(sigma, t), planets={}, catalogue_longitudes={'Jupiter': 0.5})
    series.validate()
    assert owner.catalogue_planets(series) == ('Jupiter',)
    line = 0.5 + np.radians(const.PLANET_N_DEG_PER_GAUSS_YR['Jupiter']) * (t - t[0])
    assert np.allclose(owner.catalogue_line(series, 'Jupiter'), line, atol=1e-9)
    assert np.allclose(np.mod(owner.planet_longitude(series, 'Jupiter'), TWO_PI), np.mod(line, TWO_PI), atol=1e-9)


@pytest.mark.parametrize('bad', [{'Pluto': 0.1}, {'Jupiter': float('nan')}])
def test_validate_rejects_bad_catalogue_longitudes(bad):
    t = np.arange(0.0, 1000.0, 10.0)
    series = dataclasses.replace(hilda_record(np.pi + 0.5 * np.sin(TWO_PI * t / 250.0), t), planets={}, catalogue_longitudes=bad)
    with pytest.raises(ValueError):
        series.validate()
