"""Independent functional tests of the W6 verdict changes (C1/C6, C2, C3, C5).

Every expected value is derived here, not read from the code:

- coverage (C3): hand-built turning points with known times; the gaps, the longest TP-to-TP step and the trimmed
  stretch are computed by hand from the times, and the rule of docs/libration-analysis.md (G2) is applied by hand;
- angle_status / final_regime (C1/C6): synthetic records whose angle rate is written down analytically (a harmonic
  libration, a uniform circulation, smooth switches between them); the body's a and lambda are built so that the
  identity sigma_dot = 3 n_J - n(a) holds exactly. The turning points are the zeros of the analytic rate, found
  here by root finding;
- G2c retry (C5): a slow libration with a faster wave on top; the expected cutoffs are 3 x the wave period (W_c)
  and half the slow period (the retry);
- W and S margins (C2): a synthetic 1:1 tadpole for W; hand-built scan results for S, with the ratios computed here.
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from resonances.resonance.analysis import geometry as G
from resonances.resonance.analysis.ledger import Ledger, Margin
from resonances.resonance.analysis.mmr import context, decide, owner
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.models import (
    MMRSeries,
    OwnerResult,
    Regime,
    ResonanceSpec,
    SubresonanceCandidate,
)
from resonances.resonance.analysis.mmr.subresonances import SubresonanceScan
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import (
    AngleStatus,
    AngleSubtype,
    CheckValue,
    Reason,
    ResonanceType,
    Status,
    Verdict,
)

SETTINGS = AnalysisSettings()
TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------- helpers: coverage on hand-built turning points

RECORD_T = np.arange(0.0, 10000.1, 1.0)  # record [0, 10000] yr, step 1 yr


def _episode(times, window):
    """A run of alternating TPs (max first) at `times`, half-amplitude 1 rad about pi, classified on RECORD_T."""
    times = np.asarray(times, dtype=float)
    kinds = np.array([1 if i % 2 == 0 else -1 for i in range(len(times))])
    tp = G.TurningPoints(
        indices=np.searchsorted(RECORD_T, times),
        times_yr=times,
        values_rad=np.pi + kinds * 1.0,
        kinds=kinds,
        prominence_rad=np.radians(1.0),
    )
    return G.classify_run(0, tp, np.arange(len(times)), RECORD_T, window, SETTINGS)


def _expected_cover(times, window, record=(0.0, 10000.0), coverage=1.0):
    """The rule of the docs, by hand: (start, end) covered and every number it reads."""
    times = np.asarray(times, dtype=float)
    period = float(np.median(times[2:] - times[:-2]))
    step_max = float(np.max(np.diff(times)))
    out = {}
    for edge, gap_rec, gap_win, hidden in (
        ('start', times[0] - record[0], times[0] - window[0], window[0] - record[0]),
        ('end', record[1] - times[-1], window[1] - times[-1], record[1] - window[1]),
    ):
        old = gap_rec / period <= coverage
        new = gap_win <= step_max and hidden / period <= coverage
        out[edge] = dict(gap_p=gap_rec / period, gap_yr=gap_win, hidden_p=hidden / period, covered=old or new)
    return period, step_max, out


def _check_numbers(ep, times, window):
    period, step_max, exp = _expected_cover(times, window)
    assert ep.accepted
    assert ep.period_med_yr == pytest.approx(period)
    assert ep.tp_step_max_yr == pytest.approx(step_max)
    assert ep.coverage_start_gap_p == pytest.approx(exp['start']['gap_p'])
    assert ep.coverage_end_gap_p == pytest.approx(exp['end']['gap_p'])
    assert ep.window_start_gap_yr == pytest.approx(exp['start']['gap_yr'])
    assert ep.window_end_gap_yr == pytest.approx(exp['end']['gap_yr'])
    assert ep.window_start_hidden_p == pytest.approx(exp['start']['hidden_p'])
    assert ep.window_end_hidden_p == pytest.approx(exp['end']['hidden_p'])
    return exp['start']['covered'], exp['end']['covered']


def test_coverage_new_rule_covers_an_edge_the_record_gap_misses():
    # P = 1000 yr; first TP 1.1 P after the record start (old rule fails), 300 yr after the window start (<= 500 yr
    # step) with 0.8 P trimmed: covered by the new rule. End: 0.9 P from the record's end, old rule.
    times = 1100.0 + 500.0 * np.arange(17)
    window = (800.0, 9200.0)
    ep = _episode(times, window)
    expected = _check_numbers(ep, times, window)
    assert expected == (True, True)
    assert G.covers_edges(ep, SETTINGS) == expected
    assert G.covering_episode([ep], SETTINGS) is ep


def test_coverage_trim_hiding_several_periods_does_not_cover():
    # 12499-like: the window starts 2.5 P after the record (a long cutoff); no TP missed inside the window, but the
    # trim hides 2.5 P, so neither edge is covered.
    times = 2700.0 + 500.0 * np.arange(10)
    window = (2500.0, 7500.0)
    ep = _episode(times, window)
    expected = _check_numbers(ep, times, window)
    assert expected == (False, False)
    assert G.covers_edges(ep, SETTINGS) == expected
    assert G.covering_episode([ep], SETTINGS) is None


def test_coverage_missing_turning_point_at_window_edge_does_not_cover():
    # First TP 700 yr after the window start: longer than the 500-yr step, a TP is missing; record gap 1.5 P.
    # End: exactly 1.0 P from the record's end (old rule, <=).
    times = 1500.0 + 500.0 * np.arange(16)
    window = (800.0, 9200.0)
    ep = _episode(times, window)
    expected = _check_numbers(ep, times, window)
    assert expected == (False, True)
    assert G.covers_edges(ep, SETTINGS) == expected
    assert G.covering_episode([ep], SETTINGS) is None


def test_coverage_compares_with_the_longest_step_not_the_median():
    # Uneven half-cycles 400/600 yr (full period still 1000 yr). Window gap 550 yr: above the median step (500)
    # but within the longest (600): covered.
    steps = np.tile([400.0, 600.0], 10)[:15]
    times = 1350.0 + np.concatenate([[0.0], np.cumsum(steps)])
    window = (800.0, 9200.0)
    ep = _episode(times, window)
    expected = _check_numbers(ep, times, window)
    assert np.median(np.diff(times)) < 550.0 <= np.max(np.diff(times))
    assert expected == (True, True)
    assert G.covers_edges(ep, SETTINGS) == expected


def test_coverage_old_rule_kept_when_the_window_gap_is_long():
    # No trim (window = record): the first TP at 950 yr is a whole 950 yr from the window start (> 500-yr step),
    # but within 1 P of the record start: the old rule still covers it.
    times = 950.0 + 500.0 * np.arange(18)
    window = (0.0, 10000.0)
    ep = _episode(times, window)
    expected = _check_numbers(ep, times, window)
    assert expected == (True, True)
    assert G.covers_edges(ep, SETTINGS) == (True, True)


# ---------------------------------------------------------------- helpers: synthetic 3J-1 records

N_J = context.table_mean_motion('Jupiter')
STEP = 2.0
T = np.arange(0.0, 30000.0 + 0.1, STEP)


def _smooth_step(t, t0, t1):
    x = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * x))


def _series_3j1(t, rate, name='synthetic'):
    """3J-1: sigma = 3 lambda_J - lambda - 2 varpi. With varpi fixed, n = lambda_dot = 3 n_J - sigma_dot exactly."""
    spec = ResonanceSpec.from_key('3J-1')
    sigma = np.concatenate([[np.pi], np.pi + np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))])
    varpi = np.full_like(t, 0.8)
    lam = 3.0 * (N_J * t + 0.2) - 2.0 * varpi - sigma
    a = (TWO_PI / (3.0 * N_J - rate)) ** (2.0 / 3.0)
    return MMRSeries(
        t_yr=t,
        a=a,
        e=np.full_like(t, 0.1),
        inc=np.full_like(t, 0.05),
        Omega=np.full_like(t, 0.3),
        omega=np.full_like(t, 0.5),
        varpi=varpi,
        angle_wrapped=np.mod(sigma, TWO_PI),
        resonance=spec,
        regime=Regime(step_yr=t[1] - t[0], span_yr=t[-1] - t[0], n_rows=len(t), integrator='synthetic'),
        mean_longitude=np.mod(lam, TWO_PI),
        mean_anomaly=np.mod(lam - varpi, TWO_PI),
        name=name,
    )


def _analyse(rate, t=T):
    return analyse_mmr(_series_3j1(t, rate), SETTINGS, scan_subresonances=False, scan_owner=False)


def _zeros(rate_fn, t0, t1, dt=1.0):
    """Zeros of the analytic rate (the turning points of sigma), by sign change and Brent's method."""
    grid = np.arange(t0, t1 + dt, dt)
    values = rate_fn(grid)
    roots = []
    for i in np.nonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0)[0]:
        roots.append(brentq(rate_fn, grid[i], grid[i + 1]))
    return np.array(roots)


LIB_A, LIB_P = 1.0, 1000.0  # libration half-amplitude [rad], period [yr]
CIRC_RATE = TWO_PI / 1500.0  # circulation: one turn per 1500 yr


def _lib(t):
    return LIB_A * (TWO_PI / LIB_P) * np.cos(TWO_PI * t / LIB_P)


def test_full_libration_reports_libration_to_the_end():
    rate_fn = _lib
    analysis = _analyse(rate_fn(T))
    v = analysis.verdict
    tps = _zeros(rate_fn, T[0], T[-1])  # 250, 750, ..., 29750
    assert v.angle_status == AngleStatus.LIBRATION
    assert v.status == Status.RESONANT
    # the last (only) librating episode reaches the record's end: final_regime libration from its start (first TP)
    assert v.final_regime == AngleStatus.LIBRATION
    assert v.final_regime_start_yr == pytest.approx(tps[0], abs=5.0)


def test_pure_circulation_has_no_final_regime():
    v = _analyse(np.full_like(T, CIRC_RATE)).verdict
    assert v.angle_status == AngleStatus.CIRCULATION
    assert v.final_regime == ''
    assert v.final_regime_start_yr is None


def test_g1c_one_episode_then_circulation_is_transient_ending_in_circulation():
    def rate_fn(t):
        s = _smooth_step(t, 12000.0, 13000.0)
        return _lib(t) * (1.0 - s) + CIRC_RATE * s

    v = _analyse(rate_fn(T)).verdict
    last_tp = _zeros(rate_fn, T[0], T[-1])[-1]
    assert 12000.0 < last_tp < 13000.0
    assert v.angle_status == AngleStatus.EPISODIC  # C1: no longer 'circulation'
    assert v.angle_subtype in (AngleSubtype.NONE, '')  # one episode: not recurrent
    assert v.final_regime == AngleStatus.CIRCULATION
    assert v.final_regime_start_yr == pytest.approx(last_tp, abs=10.0)  # G1c: from the last TP
    assert v.status == Status.TRANSIENT


def test_g1c_two_episodes_then_circulation_is_recurrent():
    def rate_fn(t):
        w = _smooth_step(t, 8000.0, 9000.0) - _smooth_step(t, 14000.0, 15000.0) + _smooth_step(t, 22000.0, 23000.0)
        return _lib(t) * (1.0 - w) + CIRC_RATE * w

    analysis = _analyse(rate_fn(T))
    v = analysis.verdict
    accepted = [e for e in analysis.geometry.runs if e.accepted]
    assert len(accepted) == 2
    last_tp = _zeros(rate_fn, T[0], T[-1])[-1]
    assert v.angle_status == AngleStatus.EPISODIC
    assert v.angle_subtype == AngleSubtype.RECURRENT
    assert v.final_regime == AngleStatus.CIRCULATION
    assert v.final_regime_start_yr == pytest.approx(last_tp, abs=10.0)


def test_capture_into_libration_ends_in_libration():
    def rate_fn(t):
        s = _smooth_step(t, 12000.0, 13000.0)
        return CIRC_RATE * (1.0 - s) + _lib(t) * s

    v = _analyse(rate_fn(T)).verdict
    first_tp = _zeros(rate_fn, T[0], T[-1])[0]  # circulation has no TP: the first zero opens the episode
    assert 12000.0 < first_tp < 13000.0
    assert v.angle_status == AngleStatus.EPISODIC
    assert v.final_regime == AngleStatus.LIBRATION
    assert v.final_regime_start_yr == pytest.approx(first_tp, abs=10.0)


def test_g1e_libration_then_jump_of_a_into_circulation():
    # A small libration (0.3 rad, 2000 yr) and an abrupt switch (20 yr) to a fast circulation at 15 kyr: a jumps by
    # (2 pi / 300 yr) / |c| ~ 0.02 au. The last regime starts tau/2 after the jump, tau = max(2 P_c, 5 dt);
    # P_c = max(1.5 P_syn, P_floor / 2) (W16: from the angle's physics only) with P_syn = 2 pi / |n - n_J| =
    # 2 pi / (2 n_J) ~ 5.9 yr and P_floor = 150 yr (the shortest libration period of an ordinary MMR): P_c = 75 yr,
    # tau = 150 yr. The increment over tau of a step spread over ~P_c is flat for |t - t_jump| < (tau - P_c) / 2, so
    # the locator places it within that much (36 yr here; 30 yr was the slack at tau = 80 yr).
    t_jump, amp, period, fast = 15000.0, 0.3, 2000.0, TWO_PI / 300.0

    def rate_fn(t):
        s = _smooth_step(t, t_jump - 10.0, t_jump + 10.0)
        return amp * (TWO_PI / period) * np.cos(TWO_PI * t / period) * (1.0 - s) + fast * s

    p_syn = TWO_PI / (2.0 * N_J)
    p_c = max(1.5 * p_syn, 150.0 / 2.0)
    tau = max(2.0 * p_c, 5.0 * STEP)
    assert p_c == pytest.approx(75.0)

    v = _analyse(rate_fn(T)).verdict
    last_tp = _zeros(rate_fn, T[0], t_jump + 10.0)[-1]
    assert v.angle_status == AngleStatus.EPISODIC
    assert v.final_regime == AngleStatus.CIRCULATION
    # G1e: from the start of the last regime (after the jump), later than the last TP (which G1c would give)
    assert v.final_regime_start_yr > last_tp
    assert t_jump <= v.final_regime_start_yr <= t_jump + tau / 2.0 + (tau - p_c) / 2.0
    # the episode librates about pi, outside the well of the directly averaged potential (centre_outside_well): with
    # L4 and R0 not measured a failed strength check is a measured objection (rule 5o, W18)
    assert (v.status, v.reason) == (Status.UNCERTAIN, Reason.ORIGIN_UNKNOWN)


def test_final_regime_columns_round_trip_and_order():
    v = Verdict(
        resonance_type=ResonanceType.MMR,
        status=Status.TRANSIENT,
        angle_status=AngleStatus.EPISODIC,
        angle_subtype=AngleSubtype.RECURRENT,
        final_regime=AngleStatus.CIRCULATION,
        final_regime_start_yr=12554.0,
    )
    row = v.to_flat_dict()
    keys = list(row)
    i = keys.index('angle_subtype')
    assert keys[i + 1 : i + 3] == ['final_regime', 'final_regime_start_yr']
    assert row['final_regime'] == 'circulation'
    assert row['final_regime_start_yr'] == pytest.approx(12554.0)
    back = Verdict.from_flat_dict(row)
    assert back.final_regime == AngleStatus.CIRCULATION
    assert back.final_regime_start_yr == pytest.approx(12554.0)

    empty = Verdict(resonance_type=ResonanceType.MMR, status=Status.NON_RESONANT, angle_status=AngleStatus.CIRCULATION)
    row = empty.to_flat_dict()
    assert row['final_regime'] == '' and row['final_regime_start_yr'] is None
    back = Verdict.from_flat_dict(row)
    assert back.final_regime == '' and back.final_regime_start_yr is None


def test_summary_row_carries_final_regime_after_angle_subtype():
    def rate_fn(t):
        s = _smooth_step(t, 12000.0, 13000.0)
        return _lib(t) * (1.0 - s) + CIRC_RATE * s

    row = _analyse(rate_fn(T)).summary_row()
    keys = list(row)
    i = keys.index('angle_subtype')
    assert keys[i + 1 : i + 3] == ['final_regime', 'final_regime_start_yr']
    assert row['angle_status'] == 'episodic' and row['final_regime'] == 'circulation'


# ---------------------------------------------------------------- C5: G2c retry at P / FILTER_MIN_LIBRATION_CUTOFFS

T_LONG = np.arange(0.0, 200000.0 + 0.1, 20.0)


def _slow_with_wave(slow_amp, slow_period, wave_amp, wave_period):
    return slow_amp * (TWO_PI / slow_period) * np.cos(TWO_PI * T_LONG / slow_period) + wave_amp * (TWO_PI / wave_period) * np.cos(
        TWO_PI * T_LONG / wave_period
    )


def test_g2c_retry_at_half_the_libration_period():
    # Slow libration 2 rad, 21.6 kyr (centre range 4 rad > pi: the fine runs' centre moves, no fine libration);
    # waves 0.5 rad, 4.1 kyr dominate the rate, so P_cyc ~ 4.1 kyr and W_c = 3 P_cyc ~ 12.3 kyr. The libration at
    # W_c has P = 21.6 kyr < 2 W_c: too slow a filter; the retry at P / 2 = 10.8 kyr decides libration again.
    slow_p, wave_p = 21600.0, 4100.0
    analysis = analyse_mmr(
        _series_3j1(T_LONG, _slow_with_wave(2.0, slow_p, 0.5, wave_p)),
        SETTINGS,
        scan_subresonances=False,
        scan_owner=False,
    )
    v = analysis.verdict
    cols = analysis.extra_columns
    first = cols['geo_centre_scale_first_cutoff_yr']
    assert first == pytest.approx(3.0 * wave_p, rel=0.1)
    assert slow_p < SETTINGS.filter_min_libration_cutoffs * first  # the precondition of the retry
    assert cols['geo_centre_scale_cutoff_yr'] == pytest.approx(slow_p / SETTINGS.filter_min_libration_cutoffs, rel=0.05)
    assert cols['geo_centre_scale_basis'] == 'period_cycle'
    assert v.angle_status == AngleStatus.LIBRATION
    assert any('W_c too slow' in e.name for e in v.decision_path)


def test_g2c_without_retry_when_the_libration_is_slow_enough():
    # Smaller waves: P_cyc comes out shorter, W_c ~ 9.8 kyr and 2 W_c < 21.6 kyr: no filter_too_slow, no retry.
    slow_p = 21600.0
    analysis = analyse_mmr(
        _series_3j1(T_LONG, _slow_with_wave(2.0, slow_p, 0.3, 4100.0)),
        SETTINGS,
        scan_subresonances=False,
        scan_owner=False,
    )
    cols = analysis.extra_columns
    if 'geo_centre_scale_cutoff_yr' not in cols:
        pytest.skip('G2c did not run on this record')
    w_c = cols['geo_centre_scale_cutoff_yr']
    first = cols.get('geo_centre_scale_first_cutoff_yr', float('nan'))
    # the rule: a retry exactly when P < FILTER_MIN_LIBRATION_CUTOFFS x W_c
    assert slow_p >= SETTINGS.filter_min_libration_cutoffs * w_c
    assert first is None or np.isnan(first)
    assert not any('W_c too slow' in e.name for e in analysis.verdict.decision_path)


# ---------------------------------------------------------------- C2: W and S margins


def test_w_margin_not_applicable_for_a_coorbital_tadpole():
    # 1E-1: sigma = lambda_E - lambda; a 300-yr tadpole of 0.3 rad about 60 deg, n = n_E - sigma_dot exactly.
    spec = ResonanceSpec.from_key('1E-1')
    n_e = context.table_mean_motion('Earth')
    t = np.arange(0.0, 2000.0 + 0.01, 0.05)
    period, amp = 300.0, 0.3
    sigma = np.pi / 3.0 + amp * np.sin(TWO_PI * t / period)
    rate = amp * (TWO_PI / period) * np.cos(TWO_PI * t / period)
    lam = n_e * t - sigma
    varpi = np.full_like(t, 1.0)
    series = MMRSeries(
        t_yr=t,
        a=(TWO_PI / (n_e - rate)) ** (2.0 / 3.0),
        e=np.full_like(t, 0.05),
        inc=np.full_like(t, 0.02),
        Omega=np.full_like(t, 0.3),
        omega=np.full_like(t, 0.7),
        varpi=varpi,
        angle_wrapped=np.mod(sigma, TWO_PI),
        resonance=spec,
        regime=Regime(step_yr=0.05, span_yr=2000.0, n_rows=len(t), integrator='synthetic'),
        mean_longitude=np.mod(lam, TWO_PI),
        mean_anomaly=np.mod(lam - varpi, TWO_PI),
        name='coorbital',
    )
    analysis = analyse_mmr(series, SETTINGS, scan_subresonances=False, scan_owner=True)
    assert owner.FLAG_COORBITAL in analysis.owner.flags
    assert analysis.verdict.angle_status == AngleStatus.LIBRATION
    assert analysis.verdict.margins['W'].margin == Margin.NOT_APPLICABLE


def _owner_result(flags, check):
    return OwnerResult(
        n_candidates=0,
        candidates=[],
        accepted=None,
        check=check,
        sigma_r2=np.nan,
        psi_rate_rad_yr=np.nan,
        psi_slow_corr=np.nan,
        psi_varpi_corr=np.nan,
        freq_r2_owner=np.nan,
        freq_r2_sigma=np.nan,
        freq_period_yr=np.nan,
        flags=flags,
    )


def test_w_margin_retrograde_stays_not_measured():
    result = _owner_result([owner.FLAG_RETROGRADE], CheckValue.NOT_MEASURED)
    margin = decide._w_margin(result, Ledger.from_settings(SETTINGS), SETTINGS)
    assert margin.margin == Margin.NOT_MEASURED


def _candidate(key, centre_range, accepted=False):
    return SubresonanceCandidate(
        spec=ResonanceSpec.from_key(key),
        j=0,
        nodal=False,
        centre_range_rad=centre_range,
        centre_rate_rad_yr=0.0,
        centre_max_pass_rad=centre_range,
        centre_mean_rad=0.0,
        phi0_coherence=0.5,
        coefficient_rel=1.0,
        composite_rms_rad=np.nan,
        period_corr=np.nan,
        period_ratio=np.nan,
        rank=1,
        accepted=accepted,
        reason='' if accepted else 'centre_moves',
    )


def _scan(candidates, centre0_range, accepted=None, theory=False, tie=False):
    return SubresonanceScan(
        triggers=(),
        support=(),
        candidates=candidates,
        accepted=accepted,
        check=CheckValue.PASS,
        composite=False,
        undetermined=False,
        tie_by_order=tie,
        theory_measured=theory,
        sigma0_phi0_coherence=np.nan,
        composite_rms_rad=np.nan,
        single_rms_rad=np.nan,
        binomial_mode=0,
        drift_pred_rad_yr=np.nan,
        drift_meas_rad_yr=np.nan,
        plane_offset_rad=np.nan,
        centre0_range_rad=centre0_range,
    )


@pytest.mark.parametrize(
    'ranges, centre0, expected_ratio',
    [
        ([2.0, 3.0], 2.0, 1.0),  # min 2.0 / 2.0
        ([1.7, 2.4], 2.0, 0.85),  # min 1.7 / 2.0 > 0.8: full
        ([1.5, 2.6], 2.0, 0.75),  # min 1.5 / 2.0 <= 0.8: a near miss, thin
    ],
)
def test_s_margin_rejected_scan_graded_by_closest_member(ranges, centre0, expected_ratio):
    cands = [_candidate(key, r) for key, r in zip(['9M-16+1+6', '9M-16+2+5'], ranges)]
    margin = decide._s_margin(_scan(cands, centre0), Ledger.from_settings(SETTINGS), SETTINGS)
    assert margin.value == pytest.approx(expected_ratio)
    expected = Margin.FULL if expected_ratio > SETTINGS.mmr_s_close_ratio else Margin.THIN
    assert margin.margin == expected


def test_s_margin_rejected_scan_without_a_range_is_full():
    for cands, centre0 in (([], 2.0), ([_candidate('9M-16+1+6', 1.0)], np.nan)):
        margin = decide._s_margin(_scan(cands, centre0), Ledger.from_settings(SETTINGS), SETTINGS)
        assert margin.margin == Margin.FULL


def test_s_margin_accepted_member_thin_only_by_tie_or_missing_theory():
    member = _candidate('9M-16+1+6', 0.5, accepted=True)
    ledger = Ledger.from_settings(SETTINGS)
    confirmed = decide._s_margin(_scan([member], 2.0, accepted=member, theory=True), ledger, SETTINGS)
    no_theory = decide._s_margin(_scan([member], 2.0, accepted=member, theory=False), ledger, SETTINGS)
    tie = decide._s_margin(_scan([member], 2.0, accepted=member, theory=True, tie=True), ledger, SETTINGS)
    assert confirmed.margin == Margin.FULL
    assert no_theory.margin == Margin.THIN
    assert tie.margin == Margin.THIN
