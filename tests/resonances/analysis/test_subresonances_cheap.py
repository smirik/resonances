"""The cheap subresonance numbers of every row and the triggers T7/T8 (W28 A6a; cards g06-8a A, g06-8b A, g06-2, g06-4).

Independent truths: |P(beta)| of two members C_0 = 1, C_1 = 0.9 by the law of cosines, |1 + 0.9 e^{i beta}| =
sqrt(1.81 + 1.8 cos beta), and its argument atan2(0.9 sin beta, 1 + 0.9 cos beta) (monotone for |beta| < 154 deg);
centre tracks built from linear slow angles, whose member offsets are known exactly (Delta of 9M-16+j+(7-j) is
j (varpi_M - varpi)); the shift and half-amplitude of 2003 ND (61 and 55 deg) and 2003 EH16 (21 and 19 deg) read from
their episodes.csv rows. Benchmark checks: 12218 4M-7 (the gap is the cancellation of the compound term), 2003 UG22
40 kyr (the flattest planar frames are e_Earth^5-6 terms, 'a frame, not a force'), ND and EH16 (T8 diagnostic only)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from resonances.resonance.analysis.geometry import EpisodeShape
from resonances.resonance.analysis.mmr import io, subresonances as sr
from resonances.resonance.analysis.mmr.models import (
    DIAGNOSTIC_TRIGGERS,
    MMRSeries,
    PlanetSeries,
    Preprocessed,
    Regime,
    ResonanceSpec,
    SubresonanceTrigger,
)
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import CheckValue, Reason
from tests.resonances.analysis import builders

S = settings_from_config()
NAN = float('nan')
TWO_PI = 2 * math.pi


def _pair_magnitude(beta: float) -> float:
    """|1 + 0.9 e^{i beta}| by the law of cosines."""
    return math.sqrt(1.81 + 1.8 * math.cos(beta))


def _pair_argument(beta: float) -> float:
    return math.atan2(0.9 * math.sin(beta), 1 + 0.9 * math.cos(beta))


# --- the theory numbers on hand-built inputs ---------------------------------------------------------------------


def _pair_inputs():
    """beta from 0 to 2 pi over 1000 yr; C_0 = 1, C_1 = 0.9 constant; episodes 0-300 yr and 700-1000 yr."""
    t = np.linspace(0.0, 1000.0, 2001)
    beta = TWO_PI * t / 1000.0
    return t, np.array([1.0 + 0j, 0.9 + 0j]), [0, 1], beta


def test_theory_numbers_put_the_minimum_of_the_compound_term_in_the_gap():
    t, coefficients, js, beta = _pair_inputs()
    numbers = sr.theory_numbers(t, coefficients, js, beta, 0, [0, 1], [(0.0, 300.0, 0.5), (700.0, 1000.0, 1.0)])
    # beta = pi at t = 500 yr, inside the gap 300-700 yr: |P| = 0.1 of max 1.9
    assert numbers.p_min_gap == pytest.approx(0.1 / 1.9, rel=1e-6)
    assert numbers.p_min_gap < 0.15
    edges = dict(numbers.p_edges)
    assert edges[300.0] == pytest.approx(_pair_magnitude(0.6 * math.pi) / 1.9, rel=1e-6)  # 0.5893
    assert edges[700.0] == pytest.approx(_pair_magnitude(1.4 * math.pi) / 1.9, rel=1e-6)
    assert edges[0.0] == pytest.approx(1.0)
    assert dict(numbers.shares) == pytest.approx({0: 1 / 1.9, 1: 0.9 / 1.9})
    assert numbers.leading_j == 0 and numbers.rouche is True  # |C_0| = 1 > 0.9
    # the catalogue member j = 0: its centre pi - arg P moves by atan2(0.9 sin 108 deg, 1 + 0.9 cos 108 deg) in each
    # episode; the first has half-amplitude 0.5 rad, the second 1.0 rad
    travel = _pair_argument(0.6 * math.pi)
    assert numbers.centre_travel_rad == pytest.approx(travel, rel=1e-4)
    assert numbers.travel_ratio == pytest.approx(travel / 0.5, rel=1e-4)


def test_theory_numbers_of_the_other_member_and_without_episodes():
    t, coefficients, js, beta = _pair_inputs()
    # catalogue member j = 1: c = pi - arg P + beta; Rouche fails for it (0.9 < 1)
    numbers = sr.theory_numbers(t, coefficients, js, beta, 1, [0, 1], [(0.0, 300.0, 1.0)])
    assert numbers.rouche is False
    assert numbers.centre_travel_rad == pytest.approx(0.6 * math.pi - _pair_argument(0.6 * math.pi), rel=1e-4)
    # no episode: the shares and Rouche are there, the episode numbers are not
    bare = sr.theory_numbers(t, coefficients, js, beta, 0, [0, 1], [])
    assert bare.shares and bare.rouche is True
    assert math.isnan(bare.p_min_gap) and math.isnan(bare.centre_travel_rad) and bare.p_edges == ()


# --- triggers T7 and T8 --------------------------------------------------------------------------------------------


def _moving_run(shift_deg: float, half_deg: float, overlap: bool = True):
    return builders.make_episode(
        4,
        shape=EpisodeShape.STATIONARY,
        bands_overlap=overlap,
        centre_shift_rad=math.radians(shift_deg),
        half_amp_med_rad=math.radians(half_deg),
    )


def _quiet():
    return builders.make_response(phi0_coherence=0.9)


def test_t8_fires_on_a_moving_centre_with_standing_bands_and_on_origin_unknown():
    def found(runs, reason=''):
        return sr.triggers(builders.make_geometry(runs=runs), _quiet(), reason, 0, True, S)[0]

    assert found([_moving_run(61.3, 54.8)]) == (SubresonanceTrigger.T8,)  # 2003 ND 6J-6S-1
    assert found([_moving_run(-21.1, 18.7)]) == (SubresonanceTrigger.T8,)  # 2003 EH16, a negative shift
    assert found([_moving_run(40.0, 54.8)]) == ()  # the centre moves less than the half-amplitude
    assert found([_moving_run(61.3, 54.8, overlap=False)]) == ()  # the bands do not overlap: a moving run, not T8
    assert found([], str(Reason.ORIGIN_UNKNOWN)) == (SubresonanceTrigger.T8,)
    assert SubresonanceTrigger.T8 in DIAGNOSTIC_TRIGGERS and SubresonanceTrigger.T7 not in DIAGNOSTIC_TRIGGERS


def test_t7_fires_when_the_theory_centre_moves_more_than_the_half_amplitude():
    def found(ratio):
        return sr.triggers(builders.make_geometry(), _quiet(), '', 0, True, S, ratio)[0]

    ratio = S.mmr_subresonance_centre_move_ratio
    assert found(2.6) == (SubresonanceTrigger.T7,)  # 12218 4M-7: 220 deg over a 85-deg half-amplitude
    assert found(0.5 * ratio) == ()
    assert found(NAN) == ()  # no theory (three-body, crossing): no T7


# --- the cheap list and the centre pass on a synthetic multiplet -------------------------------------------------


def _drifting_series(n: int = 2001, span: float = 10000.0, sigma_period_yr: float = 200.0):
    """9M-16+0+7 whose beta = varpi_M - varpi makes one turn over the record (varpi fixed); the angle circulates
    with `sigma_period_yr` (lambda follows from it: 16 lambda = 9 lambda_M + 7 varpi - sigma)."""
    spec = ResonanceSpec.from_key('9M-16+0+7')
    t = np.linspace(0.0, span, n)
    varpi = np.full(n, 0.3)
    varpi_m = 2.0 + TWO_PI * t / span
    lam_m = 1.1 + 3.3407 * t
    lam = (9 * lam_m + 7 * varpi - TWO_PI * t / sigma_period_yr) / 16
    planet = PlanetSeries(
        name='Mars',
        t_yr=t,
        a=np.full(n, 1.5237),
        e=np.full(n, 0.093),
        inc=np.full(n, 0.03),
        Omega=np.full(n, 0.8),
        varpi=np.mod(varpi_m, TWO_PI),
        mean_longitude=np.mod(lam_m, TWO_PI),
    )
    series = MMRSeries(
        t_yr=t,
        a=np.full(n, 2.2359),
        e=np.full(n, 0.159),
        inc=np.full(n, 0.1),
        Omega=np.full(n, 1.0),
        omega=np.mod(varpi - 1.0, TWO_PI),
        varpi=varpi,
        angle_wrapped=np.mod(9 * lam_m - 16 * lam + 7 * varpi, TWO_PI),
        resonance=spec,
        regime=Regime(step_yr=float(t[1] - t[0]), span_yr=span, n_rows=n),
        planets={'Mars': planet},
        mean_longitude=np.mod(lam, TWO_PI),
        name='synthetic',
    )
    beta = varpi_m - varpi
    return series, t, beta


def _cheap_scan(series, t, centre):
    """A scan of the cheap list on the whole record, the catalogue centre `centre` (W_c = 200 yr)."""
    geometry = builders.make_geometry(
        window=(float(t[0]), float(t[-1])), centre=builders.make_centre(t_yr=t, centre_rad=centre, window_yr=200.0)
    )
    prep = builders.make(Preprocessed, t_yr=t)
    context = builders.make(sr.Context, e_med=0.159)
    window, track = sr.cheap_window(geometry, S)
    return sr._Scan.build(series, prep, geometry, context, S, False, window=window, centre=track)


FACTORS = [0.159, 0.093, math.sin(0.05), math.sin(0.015)]  # e, e_Mars, sin(i/2) of the body and of Mars


def test_cheap_list_finds_the_member_whose_centre_stands_and_rejects_the_one_that_turns():
    series, t, beta = _drifting_series()
    # the catalogue centre drifts as -beta: member +1+6 (Delta = beta) stands at pi, +2+5 (Delta = 2 beta) turns once
    scan = _cheap_scan(series, t, np.pi - (beta - beta[0]))
    strongest = sr.strongest_factor(7, FACTORS[:2], FACTORS[2:])
    listed = sr.cheap_list(scan, FACTORS, strongest, S)
    by_key = {m.key: m for m in listed}
    best = listed[0]
    assert best.key == '9M-16+1+6' and best.reason == ''
    assert best.range_ratio == pytest.approx(0.0, abs=1e-3)
    # e^6 e_P / e^7 = e_P / e
    assert best.strength_rel == pytest.approx(0.093 / 0.159, rel=1e-6)
    turning = by_key['9M-16+2+5']
    assert turning.reason in sr.SCREEN_REASONS  # step 1 rejects it
    assert turning.range_ratio == pytest.approx(1.0, abs=0.01)  # c0 + 2 beta spans one turn, as c0 does
    assert turning.max_pass_rad == pytest.approx(TWO_PI, abs=0.02)  # its centre makes one turn in the episode
    # the e_Mars^8 member is weak against e^7: E' (strength (e_P/e)^8 / e ... below the floor)
    weak = by_key['9M-16+8-1']
    assert weak.strength_rel == pytest.approx((0.093 / 0.159) ** 8 * 0.159**2, rel=1e-6)
    assert (weak.reason == sr.REASON_E_STRENGTH) == (weak.strength_rel < S.mmr_subresonance_strength_rel_min)
    assert sum(1 for m in listed if not m.reason) == 1


def test_the_ranked_list_holds_the_centre_pass_of_each_member_in_the_episode():
    series, t, beta = _drifting_series()
    scan = _cheap_scan(series, t, np.pi - (beta - beta[0]))
    strongest = sr.strongest_factor(7, FACTORS[:2], FACTORS[2:])
    listed, _, _ = sr.rank_members(scan, FACTORS, strongest, False)
    passes = {m.key: m.centre_pass_rad for m in listed}
    assert passes['9M-16+1+6'] == pytest.approx(0.0, abs=1e-3)
    assert passes['9M-16+2+5'] == pytest.approx(TWO_PI, abs=0.02)  # about 360 deg
    # and step 1 rejects a track that makes that turn
    track = np.pi + beta - beta[0]
    kept, _, _, largest = sr.screen(track, np.pi - (beta - beta[0]), t, S)
    assert not kept and largest == pytest.approx(TWO_PI, abs=1e-6)


def test_a_row_without_episodes_gets_the_theory_numbers_and_no_scan():
    """A circulating synthetic 9M-16+0+7 (no turning points): the cheap theory and list are computed, no scan runs."""
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    series, _, _ = _drifting_series(n=8001, span=4000.0)
    analysis = analyse_mmr(series, S)
    assert analysis.verdict.angle_status == 'circulation'
    assert not len(analysis.geometry.episodes)
    assert analysis.subresonances is None
    cheap = analysis.subresonance_cheap
    assert cheap is not None and cheap.theory_flag == ''
    assert cheap.catalogue_j == 0 and len(cheap.shares) == 8 and sum(s for _, s in cheap.shares) == pytest.approx(1.0)
    assert math.isnan(cheap.travel_ratio) and math.isnan(cheap.p_min_gap)
    row = analysis.numbers_row()
    assert row['sub_th_shares'] and row['sub_cheap_n_members'] == cheap.n_members > 0
    assert row['sub_diag_triggers'] == '' and 'sub_triggers' not in row
    assert analysis.verdict.checks['check_mmr_subresonances'] == CheckValue.NOT_APPLICABLE


def test_a_diagnostic_scan_accepts_nothing():
    """run_subresonances(diagnostic=True) is the scan of the same record with nothing accepted: the member the plain scan
    accepts and its check are kept for sub_diag_* only."""
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    series, _, _ = _drifting_series(n=8001, span=4000.0)
    analysis = analyse_mmr(series, S, scan_subresonances=False, scan_owner=False)
    args = (series, analysis.prep, analysis.geometry, analysis.response, analysis.context, (SubresonanceTrigger.T8,), S)
    plain = sr.run_subresonances(*args)
    diagnostic = sr.run_subresonances(*args, diagnostic=True)
    assert diagnostic.accepted is None and diagnostic.check == CheckValue.NOT_APPLICABLE
    assert not any(c.accepted for c in diagnostic.candidates)
    assert diagnostic.would_check == str(plain.check)
    assert diagnostic.would_accept == (plain.accepted.spec.key if plain.accepted is not None else '')
    assert [c.spec.key for c in diagnostic.candidates] == [c.spec.key for c in plain.candidates]
    columns = sr.diagnostic_columns(diagnostic)
    assert columns['sub_diag_triggers'] == 'T8' and columns['sub_diag_n_scanned'] == len(plain.candidates)
    assert sr.diagnostic_columns(plain)['sub_diag_triggers'] == ''  # a plain scan writes no diagnostic


# --- benchmark records (local data, no integration) ----------------------------------------------------------------


def _analyse(example_id):
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    if not io.example_path(example_id).exists():
        pytest.skip('benchmark data not available')
    return analyse_mmr(io.load_series(example_id), S)


@pytest.mark.benchmark
def test_12218_gap_is_the_cancellation_of_the_compound_term():
    analysis = _analyse('mmr2__12218__4M-7+0+3')
    cheap = analysis.subresonance_cheap
    episodes = sorted((r.t_start_yr, r.t_end_yr) for r in analysis.geometry.episodes)
    assert episodes[0][1] == pytest.approx(48700, abs=1000) and episodes[1][0] == pytest.approx(76200, abs=1000)
    assert cheap.p_min_gap <= 0.15  # |P| / max |P| inside 48.7-76.2 kyr
    assert cheap.rouche is False and cheap.leading_j == 1  # the compound multiplet, +1+2 leads
    assert analysis.verdict.status == 'transient' and analysis.subresonances.accepted.spec.key == '4M-7+1+2'


@pytest.mark.benchmark
def test_ug22_flattest_planar_frames_are_no_force():
    analysis = _analyse('mmr2__2003_UG22__10E-17+0+7__40kyr')
    cheap = analysis.subresonance_cheap
    assert analysis.verdict.status == 'transient' and analysis.subresonances is None
    assert cheap.theory_flag == sr.FLAG_THEORY_CROSSING  # Earth-crosser: no theory
    assert cheap.window[0] < 1000 and cheap.window[1] == pytest.approx(20500, abs=1000)  # the first episode
    assert cheap.n == 0 and cheap.best_reason == sr.REASON_E_STRENGTH and cheap.best_strength_rel < 1e-3
    six = next(item for item in cheap.members.split(';') if item.startswith('10E-17+6+1:'))
    assert float(six.split(':')[2]) < 1e-3 and six.endswith(sr.REASON_E_STRENGTH)


@pytest.mark.benchmark
@pytest.mark.parametrize('example_id', ['mmr3__2003_ND__6J-6S-1+0+0+1', 'mmr3__2003_EH16__1M-3J-1+0+0+3'])
def test_t8_diagnostic_changes_no_verdict(example_id, monkeypatch):
    analysis = _analyse(example_id)
    row = analysis.numbers_row()
    assert row['sub_diag_triggers'] == 'T8' and 'subresonance_t8' in analysis.verdict.rules_fired
    assert analysis.subresonances is None and analysis.verdict.checks['check_mmr_subresonances'] == CheckValue.NOT_APPLICABLE
    original = sr.triggers
    monkeypatch.setattr(sr, 'triggers', lambda *a, **k: (tuple(t for t in original(*a, **k)[0] if t not in DIAGNOSTIC_TRIGGERS), ()))
    without = _analyse(example_id)
    a, b = analysis.verdict.to_flat_dict(), without.verdict.to_flat_dict()
    assert {k: v for k, v in a.items() if k != 'rules_fired'} == {k: v for k, v in b.items() if k != 'rules_fired'}
    # the W28 tree's uncertain (origin_unknown); W34: the minimal policy gives the geometry's resonant back at low (rule
    # F, geometry only; the author's resonant/low of 2026-11-03/04)
    assert builders.tree_verdict(analysis.verdict).status == 'uncertain'
    assert (a['status'], a['confidence']) == ('resonant', 'low')
