"""W10 rule R: the force law (L4), the identity (L3) and B1 per librating episode, and the response the status reads.

Synthetic records (tests/resonances/analysis/impl_response_pendulum_synthetic.py) give the expected values by
construction: on a pendulum record a = a* + sigma_dot / c, so a_dot = K sin(sigma - phi0) holds exactly (R^2 = 1),
sigma_dot = c (a - a*) holds exactly (slope/c = 1) and a carries the whole swing (B1 = 1). Where a is held flat, a
carries nothing (B1 = 0) and a_dot does not answer sin sigma. The benchmark records (skipped without the data) carry
the values the per-episode study measured with its own, monkey-patched code (impl/w10/per-episode/runs/rule_R.jsonl).
"""

from dataclasses import replace

import numpy as np
import pytest

from resonances.plotting import verdict_text as vt
from resonances.resonance.analysis.confidence import Cap
from resonances.resonance.analysis.mmr import decide, io
from resonances.resonance.analysis.mmr import models as m
from resonances.resonance.analysis.mmr import response as r
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, CheckValue, Flag

from tests.resonances.analysis.impl_response_pendulum_synthetic import episode_from, make, pendulum_record

S = settings_from_config()
# the response cutoff before 2026-10-16 (0.05 P_cyc): the slowest forced line of the angle still leaked into a_dot
NARROW = settings_from_config(mmr_response_cutoff_cycles=0.05)
PASS, FAIL, NM, NA = CheckValue.PASS, CheckValue.FAIL, CheckValue.NOT_MEASURED, CheckValue.NOT_APPLICABLE


def episode(episode_id=0, a_response=PASS, thin=False, identity=PASS, b1=PASS, n_tp=9, t_start=0.0, before_jump=False, r2=0.9):
    return make(
        m.EpisodeResponse,
        episode_id=episode_id,
        before_jump=before_jump,
        n_tp=n_tp,
        t_start_yr=t_start,
        t_end_yr=t_start + 1000.0,
        force_r2_h1=r2,
        force_r2_h2=r2,
        identity_corr=0.95,
        b1_ratio=1.0,
        a_response=a_response,
        a_response_thin=thin,
        identity=identity,
        a_amplitude=b1,
    )


def record_response(**given):
    """A Response whose own (pooled) checks are `given`, the rest not measured."""
    defaults = dict(a_response=FAIL, identity=PASS, a_amplitude=PASS, slow_libration=PASS, force_r2_h1=0.1, force_r2_h2=0.2)
    return make(m.Response, **{**defaults, **given})


# ------------------------------------------------------------------ the witness


@pytest.mark.parametrize(
    'a_response, thin, identity, b1, witness',
    [
        (PASS, False, NM, NM, True),  # strict L4, B1 not failing: L3 and B1 not needed
        (PASS, False, PASS, NM, True),  # strict L4: B1 only must not fail
        (PASS, False, FAIL, PASS, False),  # W11: a failed identity contradicts the pendulum (37087 1 Myr)
        (PASS, False, PASS, FAIL, False),  # B1 fails: a does not carry the swing
        (PASS, True, PASS, PASS, True),  # thin L4 with L3 and B1 on the same episode
        (PASS, True, PASS, NM, False),  # thin L4: B1 must pass, not only not fail
        (PASS, True, NM, PASS, False),  # thin L4: L3 must pass
        (FAIL, False, PASS, PASS, False),
        (NM, False, PASS, PASS, False),
        (NA, False, NA, NM, False),
    ],
)
def test_witness_truth_table(a_response, thin, identity, b1, witness):
    assert episode(a_response=a_response, thin=thin, identity=identity, b1=b1).witness is witness


def test_a_pooled_measurement_is_never_a_witness():
    assert not replace(episode(), episode_id=None).witness


# ------------------------------------------------------------------ the response the tree reads


def test_without_a_witness_the_record_stays():
    response = record_response(episode_responses=[episode(a_response=FAIL), episode(1, a_response=NM)])
    assert r.deciding_response(response, before_jump=False) is response
    assert not r.force_law_mixed(response)


def test_the_witness_with_most_turning_points_decides_and_the_earlier_breaks_a_tie():
    first = episode(0, n_tp=7, t_start=0.0, r2=0.91)
    longer = episode(1, n_tp=11, t_start=2000.0, r2=0.95)
    tie = episode(2, n_tp=11, t_start=5000.0, r2=0.97)
    deciding = r.deciding_response(record_response(episode_responses=[first, longer, tie]), before_jump=False)
    assert deciding.witness_episode_id == 1
    assert deciding.a_response == PASS and deciding.force_r2_h1 == 0.95
    assert (deciding.force_source, deciding.identity_source, deciding.b1_source) == (m.ResponseSource.WITNESS,) * 3


def test_a_not_measured_check_of_the_witness_never_replaces_a_measured_one():
    witness = episode(0, identity=NM, b1=NM)  # strict L4: a witness without L3 and B1
    response = record_response(identity=FAIL, identity_corr=0.2, a_amplitude=PASS, b1_ratio=0.8, episode_responses=[witness])
    deciding = r.deciding_response(response, before_jump=False)
    assert deciding.a_response == PASS and deciding.force_source == m.ResponseSource.WITNESS
    assert deciding.identity == FAIL and deciding.identity_corr == 0.2 and deciding.identity_source == m.ResponseSource.RECORD
    assert deciding.a_amplitude == PASS and deciding.b1_ratio == 0.8 and deciding.b1_source == m.ResponseSource.RECORD


def test_a_witness_and_a_failing_episode_are_mixed():
    response = record_response(episode_responses=[episode(0), episode(1, a_response=FAIL, n_tp=15)])
    assert r.force_law_mixed(response)
    assert r.deciding_response(response, before_jump=False).witness_episode_id == 0  # the failing one is no witness


def test_g1e_reads_the_episodes_before_the_jump_or_nothing():
    last_regime = record_response(a_response=PASS, slow_libration=FAIL)
    assert r.deciding_response(last_regime, before_jump=True) is None  # nothing was measured before the jump
    pooled = replace(episode(None, a_response=FAIL, identity=PASS, b1=NM, before_jump=True), episode_id=None)
    deciding = r.deciding_response(replace(last_regime, before_jump_response=pooled), before_jump=True)
    assert deciding.a_response == FAIL and deciding.a_amplitude == NM  # the last regime's values are not kept
    assert deciding.force_source == m.ResponseSource.BEFORE_JUMP and deciding.witness_episode_id is None
    assert deciding.slow_libration == NM  # B5 of the last regime describes no episode before the jump


def test_g1e_energy_stays_unmeasured_with_a_witness():
    angle = decide.AngleDecision(AngleStatus.CIRCULATION, AngleSubtype.NONE, 'G4', None, earlier_episode_id=3)
    response = record_response(episode_responses=[episode(3, before_jump=True)])
    deciding, energy = decide.regime_physics(angle, response, object())
    assert energy is None and deciding.witness_episode_id == 3 and deciding.a_response == PASS


# ------------------------------------------------------------------ measured on synthetic records


def two_episode_record():
    """A pendulum whose a carries the swing up to 3000 yr and is held at a* after it (sigma keeps swinging): episode 0
    is the pendulum, episode 1 a swing a does not carry."""
    rec = pendulum_record(span=6000.0)
    a_star = 4.0
    carry = 0.5 * (1 - np.tanh((rec.t - 3000.0) / 20.0))
    rec.prep.a_f = a_star + (rec.a - a_star) * carry
    rec.prep.adot_f = np.gradient(rec.prep.a_f, rec.t)
    tp = rec.geometry.turning_points
    first, second = tp.times_yr < 2900.0, tp.times_yr > 3100.0
    runs = [
        episode_from(turning_points_subset(tp, first), episode_id=0),
        episode_from(turning_points_subset(tp, second), episode_id=1),
    ]
    rec.geometry = replace(rec.geometry, runs=runs)
    return rec


def turning_points_subset(tp, keep):
    return type(tp)(tp.indices[keep], tp.times_yr[keep], tp.values_rad[keep], tp.kinds[keep], tp.prominence_rad)


def test_the_pendulum_episode_is_the_witness_and_the_flat_one_fails():
    rec = two_episode_record()
    physics = r.analyse_episodes(
        rec.prep, rec.geometry, rec.context, None, S, spec=rec.spec, sampling=rec.sampling, series=rec.series, judged_ids=[0, 1]
    )
    carried, flat = (p.response for p in physics)
    # episode 0: a = a* + sigma_dot / c exactly
    assert carried.a_response == PASS and not carried.a_response_thin
    assert min(carried.force_r2_h1, carried.force_r2_h2) == pytest.approx(1.0, abs=1e-3)
    assert carried.identity == PASS and carried.identity_slope_ratio == pytest.approx(1.0, abs=1e-2)
    assert carried.b1_ratio == pytest.approx(1.0, abs=0.05) and carried.a_amplitude == PASS
    assert carried.witness
    # episode 1: a flat while sigma swings: no force law, B1 ~ 0
    assert flat.a_response == FAIL and flat.a_amplitude == FAIL and flat.b1_ratio < S.mmr_b1_ratio_min
    assert not flat.witness
    response = r.analyse_response(rec.prep, rec.slow, rec.geometry, rec.context, rec.sampling, S, spec=rec.spec)
    assert response.a_response == FAIL  # pooled: the second half of the windows has no force law
    response.episode_responses = [carried, flat]
    deciding = r.deciding_response(response, before_jump=False)
    assert deciding.a_response == PASS and deciding.witness_episode_id == 0 and r.force_law_mixed(response)
    columns = r.response_columns(deciding)
    assert (columns['rsp_n_witnesses'], columns['rsp_n_episode_force_fail'], columns['rsp_witness_episode_id']) == (1, 1, 0)
    assert columns['rsp_force_source'] == 'witness'


def test_an_unjudged_run_has_numbers_and_no_checks():
    rec = pendulum_record()
    (physics,) = r.analyse_episodes(rec.prep, rec.geometry, rec.context, None, S, spec=rec.spec)
    assert physics.response.a_response is None and physics.response.identity is None and not physics.response.witness
    assert physics.response.force_r2_h1 == pytest.approx(1.0, abs=1e-3)
    assert physics.response.to_columns()['check_a_response'] == ''


def test_the_filter_guard_reads_the_episode_s_own_period():
    """P_e = 350 yr < filter_min_libration_cutoffs x P_c with P_c = 300 yr: the shape checks are not measured, their
    numbers still written; with P_c = 10 yr they are measured."""
    rec = pendulum_record()
    run = rec.geometry.runs[0]
    assert run.period_med_yr < S.filter_min_libration_cutoffs * 300.0
    rec.prep.cutoff = m.Cutoff(300.0, 'floor', 0.0, 300.0, 600.0)
    judge = r.episode_judge(run.period_med_yr, rec.prep, rec.sampling, rec.series, S)
    assert judge.too_slow
    (physics,) = r.analyse_episodes(
        rec.prep, rec.geometry, rec.context, None, S, spec=rec.spec, sampling=rec.sampling, series=rec.series, judged_ids=[0]
    )
    measured = physics.response
    assert (measured.a_response, measured.identity, measured.a_amplitude) == (NM, NM, NM)
    assert np.isfinite(measured.force_r2_h1) and measured.filter_too_slow


def test_before_jump_response_pools_the_episodes():
    rec = two_episode_record()
    runs = [replace(run, before_last_jump=True) for run in rec.geometry.runs]
    pooled = r.before_jump_response(rec.prep, rec.context, rec.sampling, S, runs, spec=rec.spec, series=rec.series)
    assert pooled.episode_id is None and pooled.before_jump and not pooled.witness
    assert pooled.n_tp == sum(run.n_tp for run in runs)
    assert pooled.period_yr == pytest.approx(np.median([run.period_med_yr for run in runs]))
    assert (pooled.t_start_yr, pooled.t_end_yr) == (runs[0].t_start_yr, runs[1].t_end_yr)
    assert pooled.a_response == FAIL  # half of its windows have no force law, as the record's pooled measurement
    assert r.before_jump_response(rec.prep, rec.context, rec.sampling, S, []) is None


# ------------------------------------------------------------------ benchmark records (the study's values)


def _facts(analysis, settings=S):
    row = {**analysis.numbers_row(), **analysis.summary_row()}
    return vt.MMRFacts(analysis, settings, row, analysis.episode_rows())


def _analyse(example, settings=S):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    return analyse_mmr(io.load_series(example), settings)


def test_33128_6n7_a_thin_witness_makes_the_transient():
    """33128 6N-7: the pooled force law fails, episode 0 passes thin with L3 and B1 (the study: uncertain ->
    transient/medium, the only status change of rule R on the benchmark). Since 2026-10-18 the mark exit_and_capture
    lifts its cap transient_unexplained (author): high."""
    analysis = _analyse('mmr2__33128__6N-7+0+1')
    v, response = analysis.verdict, analysis.response
    assert (str(v.status), str(v.libration_origin), str(v.confidence)) == ('transient', 'dynamic', 'high')
    assert response.force_source == m.ResponseSource.WITNESS and response.witness_episode_id == 0
    assert response.a_response == PASS and response.a_response_thin
    assert v.checks['check_mmr_a_response'] == PASS
    rows = {row['episode_id']: row for row in analysis.episode_rows()}
    assert rows[0]['witness'] and rows[0]['deciding_witness'] and rows[0]['check_a_response'] == 'pass'
    assert not rows[1]['deciding_witness']
    facts = _facts(analysis)  # the right column names the episode the force law and the identity were read on
    where = vt.measured_on(facts)
    assert where.startswith('measured on witness #0 (') and vt._force_detail(facts)[0].endswith(where)
    assert vt._detail_l3(facts, None)[0].endswith(vt.measured_on(facts, 'identity_source'))


def test_524522_g1e_measured_on_a_witness_before_the_jump():
    """524522 1V-1 20 kyr (G1e): the force law is measured on the episodes before the jumps; the cap
    all_physics_not_measured goes, episode_before_jump stays (the study). W18: without the libration's own image in
    the threshold the first episode before the jumps runs 52-2434 yr (34 TPs) instead of 152-477 yr (5 TPs); its
    onset O1 is then not measured (it failed on the short one) and L2, L6 are full: medium, not low."""
    analysis = _analyse('coorbital__524522__1V-1+0+0__20kyr')
    v, response = analysis.verdict, analysis.response
    assert str(v.status) == 'transient' and str(v.confidence) == 'medium'
    assert str(Cap.EPISODE_BEFORE_JUMP) in v.caps and str(Cap.ALL_PHYSICS_NOT_MEASURED) not in v.caps
    assert response.force_source == m.ResponseSource.WITNESS and response.force_law == 'coorbital'
    witness = next(e for e in response.episode_responses if e.episode_id == response.witness_episode_id)
    assert witness.before_jump and witness.witness


def test_88710_g1e_without_a_witness_reads_the_pool_before_the_jump():
    """At the old response cutoff the 9.8-yr forced line (16 P_Venus) fails the force law of the 121-cycle episode before
    the jump (R^2 0.44/0.34): no witness, the pool decides. At 0.1 P_cyc the same episode passes (0.91/0.76) and is the
    witness; the status is the same. The narrow case keeps the record's cutoff of that study, 5 yr (20 x 0.25 yr): the
    physical cutoff 75 yr removes the 9.8-yr line by itself and leaves the response cutoff nothing to show."""
    # the study's case is without the line rule of W20 CS6, which would lift the narrow cutoff above the 9.8-yr line
    old = replace(NARROW, mmr_filter_cutoff_yr=5.0, mmr_response_line_amp_frac=1e9)
    analysis = _analyse('mmr2__88710__9V-16+0+7', old)
    v, response = analysis.verdict, analysis.response
    assert (str(v.status), str(v.confidence)) == ('transient', 'medium')
    assert response.force_source == m.ResponseSource.BEFORE_JUMP and response.witness_episode_id is None
    assert not any(e.witness for e in response.episode_responses)
    n = sum(1 for e in response.episode_responses if e.before_jump)
    facts = _facts(analysis, old)
    assert vt.measured_on(facts).startswith(f'pooled over the {n} episodes before the jump (')
    assert vt._force_detail(facts)[0].endswith(vt.measured_on(facts)) and vt.earlier_regime_text(facts) == ''

    analysis = _analyse('mmr2__88710__9V-16+0+7')
    v, response = analysis.verdict, analysis.response
    assert (str(v.status), str(v.confidence)) == ('transient', 'medium')
    (witness,) = [e for e in response.episode_responses if e.witness]
    assert witness.before_jump and witness.a_response == PASS and response.force_source == m.ResponseSource.WITNESS

    # the narrow cutoff with the line rule: the 9.8-yr line (16 P_Venus) raises it to 14.8 yr and the same episode
    # is the witness, as at 0.1 P_cyc
    analysis = _analyse('mmr2__88710__9V-16+0+7', replace(NARROW, mmr_filter_cutoff_yr=5.0))
    (witness,) = [e for e in analysis.response.episode_responses if e.witness]
    assert witness.before_jump and witness.force_line_binding_yr == pytest.approx(9.84, abs=0.1)
    assert str(analysis.verdict.status) == 'transient'


def test_34706_mixed_episodes_flag_and_cap():
    """34706 7J-2 at the old response cutoff: episode 0 is a witness, another librating episode's force law fails (the
    study's conflict). That failure was the 23.7-yr forced line (2 P_Jupiter): at 0.1 P_cyc both episodes pass, the
    conflict is gone and the status stays. The narrow case keeps the record's cutoff of that study, 10 yr (20 x 0.5 yr):
    the physical cutoff 75 yr removes the 23.7-yr line by itself."""
    v = _analyse('mmr2__34706__7J-2+0-5', replace(NARROW, mmr_filter_cutoff_yr=10.0)).verdict
    assert str(Flag.EPISODE_FORCE_LAW_MIXED) in v.flags and str(Cap.EPISODE_FORCE_LAW_MIXED) in v.caps
    analysis = _analyse('mmr2__34706__7J-2+0-5')
    assert str(Flag.EPISODE_FORCE_LAW_MIXED) not in analysis.verdict.flags and analysis.verdict.status == v.status
    assert all(e.a_response != FAIL for e in analysis.response.episode_responses)
