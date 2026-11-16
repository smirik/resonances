"""The choice of the subresonance (W28 A6b, card g06-8c A): every veto before the choice, one window per episode, the
theory chooses for two-body rows, 'composite' allowed.

Independent truths: the argument of a composite term with one dominant coefficient is bounded by the triangle
inequality, |arg(1 + z)| <= asin(|z|) for |z| < 1, so the dominant member's theoretical centre c_N = pi - arg P + N beta
stays within 2 asin(sum_{j != N} |C_j| / |C_N|) while c_j of another member moves with (j - N) beta (Morbidelli 2002
§9.3; SISTERS §1.3); the numbers of test B (centre within 30 deg of 0 or pi) and of the tie band (20 %) are those of
.env.dist; the 31935 and 12218 readings of the cards g06-1 and g06-2 (the members +2+5 and +1+2 of the six accepted rows,
Claude's measurement: their theoretical centres stay within -186..-168 deg and -15..+20 deg over the episodes)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from resonances.resonance.analysis.confidence import Cap
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import decide as decide_module, roles, subresonances as sr
from resonances.resonance.analysis.mmr.models import (
    SUBRESONANCE_CHOICE_COMPOSITE,
    ResonanceSpec,
    SubresonanceCandidate,
    SubresonanceResult,
)
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import CheckValue, Flag, LibrationOrigin, OriginDetail, Reason, Status
from tests.resonances.analysis import builders

S = settings_from_config()
NAN = float('nan')
TWO_PI = 2 * math.pi


def _candidate(key, w_rad, centre_deg=180.0, reason=''):
    """An admissible-looking member (A coherent, gain over sigma0 0.6) with the measured centre range `w_rad`."""
    spec = ResonanceSpec.from_key(key)
    return SubresonanceCandidate(
        spec=spec,
        j=int(sum(spec.planet_varpi_coefficients)),
        nodal=False,
        centre_range_rad=w_rad,
        centre_rate_rad_yr=0.0,
        centre_max_pass_rad=w_rad,
        centre_mean_rad=math.radians(centre_deg),
        phi0_coherence=0.9,
        coefficient_rel=NAN,
        composite_rms_rad=NAN,
        period_corr=NAN,
        period_ratio=NAN,
        rank=1,
        accepted=False,
        reason=reason,
    )


def _evaluate(candidates, centres=None, theory=None, planar=()):
    return sr.evaluate(candidates, 0.3, 0.16, 0.1, S, theory=theory, ledger=Ledger.from_settings(S), centres=centres, planar=planar)


# --- the theoretical member centres -------------------------------------------------------------------------------


def test_the_dominant_member_has_the_flattest_theoretical_centre():
    # 9M-16-like multiplet j = 0..3 with C_2 dominant (Rouche holds: 1 > 0.1 + 0.2 + 0.15); beta sweeps half a turn
    js = [0, 1, 2, 3]
    coefficients = np.array([0.1, 0.2, 1.0, 0.15], dtype=complex)
    beta = np.linspace(0.0, math.pi, 501)
    centres = sr.theory_centres(coefficients, js, beta)
    bound = 2 * math.asin(0.45)  # the triangle-inequality bound of the range of arg(1 + z), |z| <= 0.45
    assert centres.flattest() == (2, centres.ranges[2]) and centres.ranges[2] <= bound
    for j in (0, 1, 3):
        assert centres.ranges[j] >= abs(j - 2) * math.pi - bound  # c_j moves with (j - 2) beta
    # with C_2 alone the centre stands exactly
    alone = sr.theory_centres(np.array([0, 0, 1.0, 0], dtype=complex), js, beta)
    assert alone.ranges[2] == pytest.approx(0.0, abs=1e-12) and alone.ranges[0] == pytest.approx(2 * math.pi)


# --- the choice -------------------------------------------------------------------------------------------------


def test_a_known_j_is_selected_by_the_theory_against_a_smaller_measured_range():
    flat_by_w = _candidate('9M-16+1+6', 0.50)  # the smaller measured range (W alone would take it: 0.70 > 1.2 x 0.50)
    flat_by_theory = _candidate('9M-16+2+5', 0.70)
    centres = sr.TheoryCentres({1: 3.0, 2: 0.30}, math.radians(8.0), math.radians(60.0), 3.0)
    outcome = _evaluate([flat_by_w, flat_by_theory], centres)
    assert outcome.winner is flat_by_theory and outcome.by_theory and not outcome.tie_weak
    assert outcome.reasons['9M-16+1+6'] == sr.REASON_THEORY
    # the measured centre does not follow the theory (RMS 30 > 20 deg): W is the fallback
    off = sr.TheoryCentres({1: 3.0, 2: 0.30}, math.radians(30.0), math.radians(60.0))
    outcome = _evaluate([flat_by_w, flat_by_theory], off)
    assert outcome.winner is flat_by_w and not outcome.by_theory
    # without the theory (three-body, crossing orbits): W, as before A6b
    assert _evaluate([flat_by_w, flat_by_theory]).winner is flat_by_w


def test_theoretical_ranges_within_the_tie_fraction_are_decided_by_the_measured_range():
    a = _candidate('9M-16+1+6', 0.70)
    b = _candidate('9M-16+2+5', 0.40)
    centres = sr.TheoryCentres({1: 0.30, 2: 0.33}, math.radians(8.0), math.radians(60.0))  # 0.33 - 0.30 <= 0.2 x 0.30
    outcome = _evaluate([a, b], centres)
    assert outcome.winner is b and outcome.by_theory and outcome.reasons['9M-16+1+6'] == ''


def test_a_winner_rejected_by_b_hands_the_choice_to_the_next_member():
    # before A6b test B ran on the winner only: the off-centre member (90 deg from 0 and pi, tolerance 30) won C and was
    # rejected, and nothing was accepted; now B is a veto before the choice
    off_centre = _candidate('9M-16+1+6', 0.50, centre_deg=90.0)
    on_centre = _candidate('9M-16+2+5', 0.70, centre_deg=175.0)
    assert S.mmr_subresonance_centre_tol_deg == 30.0
    outcome = _evaluate([off_centre, on_centre])
    assert outcome.winner is on_centre and outcome.reasons['9M-16+1+6'] == sr.REASON_B
    assert sr.decide([off_centre, on_centre], 0.3, 0.16, 0.1, S)[0] is on_centre
    assert sr.decide([off_centre], 0.3, 0.16, 0.1, S)[0] is None


def test_a_failed_t0_test_leaves_nothing_to_choose():
    good = _candidate('9M-16+2+5', 0.70)
    slow = sr.TheoryCheck(composite_rms_rad=math.radians(10), t0_ratio=45 / 0.94)  # 49030: 45 kyr against 0.94 kyr
    outcome = _evaluate([good, _candidate('9M-16+1+6', 0.9)], theory=slow)
    assert outcome.winner is None and set(outcome.reasons.values()) == {sr.REASON_E}


def test_two_episodes_driven_by_different_members_choose_per_episode():
    # e(t) moves the multiplet's weight from C_1 (first episode) to C_3 (second): each window's theory names its own
    t = np.linspace(0.0, 100_000.0, 4001)
    beta = TWO_PI * t / 50_000.0
    first = t < 50_000.0
    table = np.zeros((len(t), 4), dtype=complex)
    table[:, 0], table[:, 2] = 0.05, 0.1
    table[:, 1] = np.where(first, 1.0, 0.1)
    table[:, 3] = np.where(first, 0.1, 1.0)
    members = [_candidate('9M-16+1+6', 0.6), _candidate('9M-16+3+4', 0.6)]
    chosen = []
    for t0, t1 in ((5_000.0, 45_000.0), (55_000.0, 95_000.0)):
        inside = (t >= t0) & (t <= t1)
        centres = sr.theory_centres(table[inside], [0, 1, 2, 3], beta[inside], math.radians(5.0), math.radians(70.0))
        outcome = _evaluate(members, centres)
        assert outcome.by_theory
        chosen.append(outcome.winner.spec.key)
    assert chosen == ['9M-16+1+6', '9M-16+3+4']


def test_one_window_per_accepted_episode_and_the_decisive_one_gives_the_row():
    first = builders.make_episode(episode_id=0, tp_times_yr=np.arange(1000.0, 21000.0, 500.0), half_amp_med_rad=1.0)
    second = builders.make_episode(episode_id=1, tp_times_yr=np.arange(40000.0, 50000.0, 500.0), half_amp_med_rad=0.5)
    rejected = builders.make_episode(episode_id=2, tp_times_yr=np.arange(60000.0, 90000.0, 500.0), accepted=False)
    geometry = builders.make_geometry(runs=[first, second, rejected], window=(0.0, 100000.0))
    windows = sr.episode_windows(geometry, S)
    assert [w.episode_id for w in windows] == [0, 1]  # the rejected run has no window of its own
    assert windows[0].window == (1000.0, 20500.0) and windows[1].half_amplitude_rad == 0.5
    decisive, others = sr.scan_windows(geometry, S, 1)
    assert decisive.episode_id == 1 and [w.episode_id for w in others] == [0]
    decisive, others = sr.scan_windows(geometry, S, None)  # no decisive id: the longest episode window
    assert decisive.episode_id == 0
    # without an accepted episode: the longest TP-rich run (T1: a run whose centre moves), no episode id
    alone = builders.make_geometry(runs=[rejected], window=(0.0, 100000.0))
    decisive, others = sr.scan_windows(alone, S, None)
    assert decisive.episode_id is None and decisive.window == (60000.0, 89500.0) and others == []


# --- composite ----------------------------------------------------------------------------------------------------


def _pair_centres(c1, c2, rms_deg=8.0, half=0.6):
    """Members j = 1, 2 of weights c1, c2 (the rest 0) over beta in [0, 0.9 pi] (P has no zero there)."""
    beta = np.linspace(0.0, 0.9 * math.pi, 901)
    return sr.theory_centres(np.array([0, c1, c2, 0], dtype=complex), [0, 1, 2, 3], beta, math.radians(rms_deg), half)


def test_equal_weights_put_both_centres_at_half_of_beta():
    # C_1 = C_2: P = 2C cos(beta/2) e^{1.5 i beta}, so c_1 and c_2 each move by beta / 2: their ranges tie
    equal = _pair_centres(1.0, 1.0)
    assert equal.ranges[1] == pytest.approx(0.45 * math.pi, rel=1e-9) and equal.ranges[2] == pytest.approx(0.45 * math.pi, rel=1e-9)
    assert equal.beta_range_rad == pytest.approx(0.9 * math.pi)
    # C_2 = 2 C_1: |arg(1 + e^{-i beta} / 2)| <= asin(1/2), so c_2 moves by at most pi/3 < 0.45 pi
    assert _pair_centres(0.5, 1.0).ranges[2] <= math.pi / 3 + 1e-9


def test_composite_when_no_planar_member_carries_the_oscillation():
    rejected = [_candidate('9M-16+1+6', 2.0, reason=sr.REASON_RANGE), _candidate('9M-16+2+5', 2.0, reason=sr.REASON_RANGE)]
    planar = [0, 1, 2, 3]
    outcome = _evaluate(rejected, _pair_centres(1.0, 1.0), planar=planar)
    assert outcome.winner is None and outcome.composite
    assert not _evaluate(rejected, _pair_centres(0.5, 1.0), planar=planar).composite  # +2 dominates: it stands
    assert not _evaluate(rejected, _pair_centres(1.0, 1.0, rms_deg=25.0), planar=planar).composite  # not followed
    assert not _evaluate(rejected, _pair_centres(1.0, 1.0, half=3.0), planar=planar).composite  # beta 2.8 < half 3
    assert not _evaluate(rejected, _pair_centres(1.0, 1.0)).composite  # no planar members given: no reading
    # an admissible member is chosen, never 'composite'
    admissible = [_candidate('9M-16+2+5', 0.7)]
    outcome = _evaluate(admissible, _pair_centres(1.0, 1.0), planar=planar)
    assert outcome.winner is admissible[0] and not outcome.composite


def test_a_composite_regime_names_no_member_and_changes_no_status():
    scan = SubresonanceResult((), (), [], None, CheckValue.PASS, False, False, False, False, NAN, NAN, NAN, 0, NAN, NAN, NAN)
    scan.choice = SUBRESONANCE_CHOICE_COMPOSITE
    assert scan.composite_regime and scan.to_columns()['sub_choice'] == 'composite'
    own = decide_module.StatusDecision(Status.TRANSIENT, Reason.DYNAMIC_LIBRATION, LibrationOrigin.DYNAMIC, '', '')
    decide_module.composite_regime_detail(own, scan)
    assert own.status == Status.TRANSIENT and own.origin_detail == OriginDetail.COMPOSITE_MULTIPLET
    spec = ResonanceSpec.from_key('9M-16+0+7')
    found = roles.verdict_roles(spec, own.origin, own.origin_detail, own.owner, [], None, S)
    assert found.resonant_argument == roles.COMPOSITE and found.commensurability == '9M-16'
    # a circulation, or a verdict another cause owns, takes no composite detail
    circulation = decide_module.StatusDecision(Status.NON_RESONANT, Reason.CIRCULATION, LibrationOrigin.NONE, '', '')
    kinematic = decide_module.StatusDecision(Status.NON_RESONANT, Reason.KINEMATIC_LIBRATION, LibrationOrigin.KINEMATIC, '', '')
    for decision in (circulation, kinematic):
        decide_module.composite_regime_detail(decision, scan)
        assert decision.origin_detail == ''


def test_an_episode_whose_window_cannot_decide_caps_the_row():
    # 78085 5J-4S-1: the decisive episode's window passes, the first episode's members only test A could tell apart
    scan = SubresonanceResult((), (), [], None, CheckValue.PASS, False, False, False, False, NAN, NAN, NAN, 0, NAN, NAN, NAN)
    induced = decide_module.StatusDecision(Status.NON_RESONANT, Reason.INDUCED_LIBRATION, LibrationOrigin.INDUCED, '', '')
    assert decide_module.subresonance_caps(induced, scan) == []
    scan.episode_undecided = {0: [str(Flag.SUBRESONANCE_UNDETERMINED)]}
    assert decide_module.subresonance_caps(induced, scan) == [str(Cap.SUBRESONANCE_UNDETERMINED)]
    assert scan.to_columns()['sub_episode_undecided'] == '0:subresonance_undetermined'


def test_episode_arguments_column():
    scan = SubresonanceResult((), (), [], None, CheckValue.FAIL, False, False, False, False, NAN, NAN, NAN, 0, NAN, NAN, NAN)
    scan.episode_arguments = {2: '4M-7+1+2', 0: '4M-7+1+2'}
    assert scan.to_columns()['sub_episode_arguments'] == '0:4M-7+1+2;2:4M-7+1+2'


# --- benchmark: the members of 31935 and 12218 ---------------------------------------------------------------------


def _run(example_id):
    from resonances.resonance.analysis.mmr import io
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    if not io.example_path(example_id).exists():
        pytest.skip('benchmark data not available')
    return analyse_mmr(io.load_series(example_id), S)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'example_id, member, episodes',
    [
        ('mmr2__31935__9M-16+0+7', '9M-16+2+5', [(31400.0, 82300.0, '9M-16+2+5')]),  # card g06-1
        # card g06-2. In the second episode beta moves ~65 deg, less than the half-amplitude 94 deg: every member's
        # centre stays inside the oscillation, step 1 keeps none, and that episode names no argument of its own
        ('mmr2__12218__4M-7+0+3', '4M-7+1+2', [(1270.0, 48700.0, '4M-7+1+2'), (76200.0, 99800.0, '')]),
    ],
)
def test_the_theory_names_the_member_of_31935_and_12218(example_id, member, episodes):
    analysis = _run(example_id)
    scan = analysis.subresonances
    assert scan.accepted is not None and scan.accepted.spec.key == member and scan.choice == 'theory'
    assert analysis.verdict.resonant_argument == member and analysis.verdict.status == Status.TRANSIENT
    rows = sorted((row for row in analysis.episode_rows() if row['accepted']), key=lambda row: row['t_start_yr'])
    assert len(rows) == len(episodes)
    for row, (e0, e1, argument) in zip(rows, episodes):
        assert abs(row['t_start_yr'] - e0) <= 1000 and abs(row['t_end_yr'] - e1) <= 1000
        assert row['resonant_argument'] == argument


def _planar_members(example_id):
    from resonances.resonance.analysis.mmr import io, multiplet

    if not io.example_path(example_id).exists():
        pytest.skip('benchmark data not available')
    spec = io.load_series(example_id).resonance
    q = -(spec.planet_coefficients[0] + spec.body_coefficient)
    return [
        sr.member_spec(spec, (q - int(j), int(j), 0, 0))
        for j in multiplet.planar_members(spec.planet_coefficients[0], spec.body_coefficient)
    ]


@pytest.mark.benchmark
@pytest.mark.parametrize('example_id', ['mmr2__31935__9M-16+0+7', 'mmr2__12218__4M-7+0+3'])
def test_every_planar_member_as_its_own_row_gives_the_same_status_and_episodes(example_id):
    from resonances.resonance.analysis.mmr import io
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    series = io.load_series(example_id)
    parent = analyse_mmr(series, S)
    reference = sorted((r['t_start_yr'], r['t_end_yr']) for r in parent.episode_rows() if r['accepted'])
    for member in _planar_members(example_id):
        row = parent if member.key == series.resonance.key else analyse_mmr(sr.member_series(series, member), S)
        assert row.verdict.status == parent.verdict.status, member.key
        assert row.verdict.confidence == parent.verdict.confidence, member.key
        own = [(r['t_start_yr'], r['t_end_yr']) for r in row.episode_rows() if r['accepted']]
        for t0, t1 in reference:  # each of the parent's episodes is one of the member's, within 1 kyr
            assert any(abs(a - t0) <= 1000 and abs(b - t1) <= 1000 for a, b in own), (member.key, t0, t1, own)
