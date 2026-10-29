"""W20 CS7: forced-wave episodes (F6) and the kinematic drift of the slow centre (K2).

F6 (author 2026-10-19, ANSWERS Q16): an accepted episode whose period sits on a forced line of the angle's lattice
with a failed identity, or whose identity fails with the wrong sign (sigma_dot against c (a - a*)), is a forced wave of
another argument, not a libration of this angle (`decide.forced_wave_episodes`; `Geometry.forced_waves`). 37087
3J+8S-3: the 168-yr wave of 2 lambda_J - lambda - varpi (the clock read below from the saved longitudes, as in
test_impl_w11_witness), identity slope -0.35 c.

K2 (author 2026-10-19, ANSWERS Q21 and q7): a slow centre that follows the uniform lambda part plus p varpi while a does
not answer sigma, with the episodes a ripple on that drift, is non_resonant/kinematic_drift and its angle word is drift.
The synthetic record copies 21028 7E-10M-6+0+0+9: sigma = -0.0292 deg/yr t + 9 varpi + a 700-yr ripple, 9 varpi_dot
between 0.001 and 0.100 deg/yr with period 12.35 kyr. By hand: the slow centre is that sum, so its rate regressed on
9 varpi_dot has slope 1 and intercept -0.0292 deg/yr; the centre runs 50 kyr x (0.0505 - 0.0292) = 1065 deg, the
ripple's swing is 2 x 20 = 40 deg, a share of about 0.04.
"""

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import context as context_module
from resonances.resonance.analysis.mmr import decide, io
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.models import ForcedWave, MMRSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, CheckValue, LibrationOrigin, Reason, Status
from tests.resonances.analysis.test_impl_preprocess_geometry_geometry import ONE_DEG, filtered
from tests.resonances.analysis.test_impl_w2_zone_rules_decide import SAMPLING

S = settings_from_config()
FAIL, PASS, NM = CheckValue.FAIL, CheckValue.PASS, CheckValue.NOT_MEASURED
DEG = math.pi / 180


def _episode(episode_id, identity, slope, corr, line='', offset=float('nan')):
    return SimpleNamespace(
        episode_id=episode_id,
        n_tp=6,
        period_yr=168.0,
        identity=identity,
        identity_slope_ratio=slope,
        identity_corr=corr,
        force_line=line,
        force_line_offset=offset,
    )


# --- 7a: which episode is a forced wave ------------------------------------------------------------------------------


def test_forced_wave_routes():
    episodes = [
        _episode(0, FAIL, -0.36, -0.95),  # (i) the wrong sign, as tight as an identity pass needs: forced
        _episode(1, FAIL, -0.36, -0.50),  # the wrong sign but a loose relation: kept
        _episode(2, NM, -0.36, -0.95),  # the identity not measured (too short): kept
        _episode(3, FAIL, 0.72, 0.82, '3N-5', 0.4),  # (ii) on a lattice line with a failed identity: forced
        _episode(4, PASS, 1.00, 1.00, '3N-5', 0.4),  # on a line, but the identity passes: kept (482824 19N-32)
        _episode(5, PASS, 1.02, 0.99),  # a libration of this angle: kept
    ]
    waves = decide.forced_wave_episodes(episodes, S)
    assert [(w.episode_id, w.route) for w in waves] == [(0, 'wrong_sign'), (3, 'lattice_line')]
    # the author's literal "or": the line alone (MMR_FORCED_WAVE_LINE_ALONE) adds the passing episode on the line
    alone = decide.forced_wave_episodes(episodes, replace(S, mmr_forced_wave_line_alone=True))
    assert [w.episode_id for w in alone] == [0, 3, 4]


def _libration_geometry():
    """sigma = pi + 40 deg sin(2 pi t / 500 yr) over 20 kyr: one accepted episode covering the record."""
    t = np.arange(0, 20000, 1.0)
    angle = np.pi + np.radians(40) * np.sin(2 * np.pi * t / 500.0)
    return g.measure_geometry(filtered(t, angle, 50.0), ONE_DEG, S)


def test_a_forced_wave_is_no_episode_of_the_angle():
    geometry = _libration_geometry()
    assert len(decide.librating_episodes(geometry, S)) == 1
    episode = geometry.episodes[0]
    ledger = Ledger.from_settings(S)
    assert decide.decide_angle(SAMPLING, False, geometry, None, S, ledger).angle_status == AngleStatus.LIBRATION
    geometry.forced_waves = (ForcedWave(episode.episode_id, episode.n_tp, 500.0, 0.72, 0.82, '3N-5', 0.4, False, True),)
    assert geometry.episodes == [] and decide.librating_episodes(geometry, S) == []
    assert any(run.episode_id == episode.episode_id and run.accepted for run in geometry.runs)  # kept for episodes.csv
    ledger = Ledger.from_settings(S)
    angle = decide.decide_angle(SAMPLING, False, geometry, None, S, ledger)
    assert angle.angle_status not in (AngleStatus.LIBRATION, AngleStatus.EPISODIC)
    entries = {entry.name: entry for entry in ledger}
    count = entries['G2 accepted runs that are forced waves, no episodes of the angle']
    assert (count.value, count.result) == (1, False)
    line = next(e for name, e in entries.items() if 'on the lattice line 3N-5' in name)
    assert line.value == 0.4 and line.result is True
    assert decide.forced_wave_columns(geometry)['geo_forced_wave_identity_slope_offset'] == pytest.approx(0.28)


# --- 7a on the benchmark: 37087 ----------------------------------------------------------------------------------------

EXAMPLE_37087 = 'mmr3__37087__3J+8S-3+0+0-8'


def _rate(t_yr, angle):
    return float(np.polyfit(t_yr - t_yr.mean(), np.unwrap(angle), 1)[0])


@pytest.fixture(scope='module')
def analysed_37087():
    if not io.example_path(EXAMPLE_37087).exists():
        pytest.skip('benchmark data not available')
    series = io.load_series(EXAMPLE_37087)
    t = np.asarray(series.t_yr, float)
    jupiter = series.planets['Jupiter']
    psi_rate = _rate(t, series.mean_longitude) - 2 * _rate(t, jupiter.mean_longitude) + _rate(t, series.varpi)
    return analyse_mmr(series, S), 2 * np.pi / abs(psi_rate)


@pytest.mark.benchmark
def test_37087_the_168_yr_wave_is_a_forced_wave(analysed_37087):
    analysis, psi_period_yr = analysed_37087
    waves = analysis.geometry.forced_waves
    assert len(waves) == 1
    wave = waves[0]
    # the clock of 2 lambda_J - lambda - varpi from the saved longitudes (Codex W11: 166-169 yr)
    assert wave.period_yr == pytest.approx(psi_period_yr, rel=0.03)
    assert wave.on_line and wave.force_line == '2J-1' and wave.identity_slope_ratio < 0
    assert not analysis.geometry.episodes and analysis.decisive_episode_id is None
    row = next(r for r in analysis.episode_rows() if r['episode_id'] == wave.episode_id)
    assert row['accepted'] and row['forced_wave'] == 'lattice_line'


@pytest.mark.benchmark
def test_37087_without_its_forced_wave_is_non_resonant(analysed_37087):
    """The status of the author's label (non_resonant, medium) from the tree without the forced wave (reason
    circulation: the 5.7-kyr kinematic loops are no episodes at the 75-yr cutoff, the 168-yr wave's turning points fill
    the record). The angle word is his episodic (W20 round C): the forced-wave loops are librations of sigma
    geometrically, so the word is the tree's with them in, one ledger line."""
    verdict = analysed_37087[0].verdict
    assert (verdict.status, verdict.angle_status, verdict.confidence) == (Status.NON_RESONANT, AngleStatus.EPISODIC, 'medium')
    assert verdict.reason == Reason.CIRCULATION
    names = [e.name for e in verdict.decision_path]
    assert 'G2 accepted runs that are forced waves, no episodes of the angle' in names
    assert any(name.startswith('angle word episodic: sigma librates in the forced-wave runs') for name in names)
    assert verdict.orbit_type == 'symmetric forced wave about 180°, moving centre'  # W20 round D: what sigma does


def test_the_forced_wave_word_changes_no_status():
    """`forced_wave_word` on the record of `test_a_forced_wave_is_no_episode_of_the_angle`: the tree without the wave
    gives insufficient_time here, which stays (W20 round D, review C #5: only a circulation or a drift is rewritten); a
    circulation word becomes the word with the wave in, libration (the wave covers the record); without forced waves,
    or under a word the status tree set (K2), the reported word stays."""
    geometry = _libration_geometry()
    episode = geometry.episodes[0]
    geometry.forced_waves = (ForcedWave(episode.episode_id, episode.n_tp, 500.0, 0.72, 0.82, '3N-5', 0.4, False, True),)
    ledger = Ledger.from_settings(S)
    short = decide.decide_angle(SAMPLING, False, geometry, None, S, ledger)
    decision = decide.StatusDecision(Status.NON_RESONANT, Reason.CIRCULATION, LibrationOrigin.NONE, '', '')
    assert short.angle_status == AngleStatus.INSUFFICIENT_TIME
    assert decide.forced_wave_word(short, decision, SAMPLING, geometry, None, S, ledger) is short
    tree = decide.AngleDecision(AngleStatus.CIRCULATION, decide.AngleSubtype.NONE, 'C', None)
    word = decide.forced_wave_word(tree, decision, SAMPLING, geometry, None, S, ledger)
    assert word.angle_status == AngleStatus.LIBRATION
    assert geometry.forced_waves and geometry.episodes == []  # the geometry itself is untouched
    k2 = replace(decision, angle_word=AngleStatus.DRIFT)
    assert decide.forced_wave_word(tree, k2, SAMPLING, geometry, None, S, ledger) is tree
    geometry.forced_waves = ()
    assert decide.forced_wave_word(tree, decision, SAMPLING, geometry, None, S, ledger) is tree


@pytest.mark.benchmark
def test_101703_41j_14_is_unchanged():
    """A forced wave on the 3:1 line (P_e 166 yr) that the owner already explains: non_resonant/circulation."""
    example = 'mmr2__101703__41J-14+0-27'
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    verdict = analyse_mmr(io.load_series(example), S).verdict
    assert (verdict.status, verdict.angle_status, verdict.libration_origin) == (
        Status.NON_RESONANT,
        AngleStatus.CIRCULATION,
        LibrationOrigin.INDUCED,
    )


# --- 7b: the kinematic drift of the slow centre --------------------------------------------------------------------------

KEY_21028 = '7E-10M-6+0+0+9'
LAMBDA_RATE = -0.0292 * DEG  # rad/yr
P_RATE_MEAN, P_RATE_AMP, P_MOD = 0.0505 * DEG, 0.0495 * DEG, 12350.0  # 9 varpi_dot in [0.001, 0.100] deg/yr
RIPPLE_DEG, RIPPLE_YR = 20.0, 700.0


def _drift_record(a_star=2.5, c=1.0, pendulum=False) -> MMRSeries:
    t = np.arange(0.0, 50001.0, 1.0)
    p_varpi = P_RATE_MEAN * t + P_RATE_AMP * P_MOD / (2 * np.pi) * np.sin(2 * np.pi * t / P_MOD)
    ripple = RIPPLE_DEG * DEG * np.sin(2 * np.pi * t / RIPPLE_YR)
    sigma = LAMBDA_RATE * t + p_varpi + ripple
    if pendulum:  # a answers the ripple by the identity: sigma_dot = c (a - a*)
        a = a_star + np.gradient(ripple, t) / c
    else:  # a flat but for a small unrelated wave (the force law is measured, R^2 ~ 0)
        a = a_star + 2e-7 * np.sin(2 * np.pi * t / 333.0 + 0.4)
    varpi = p_varpi / 9
    zero = np.zeros_like(t)
    spec = ResonanceSpec.from_key(KEY_21028)
    return MMRSeries(t, a, np.full_like(t, 0.1), zero, zero, varpi, varpi, sigma % (2 * np.pi), spec, Regime(1.0, t[-1], len(t)))


@pytest.fixture(scope='module')
def drift_records():
    context = context_module.build_context(_drift_record(), S)
    a_star, c = context.exact_axis_au, context.identity_coefficient
    return {kind: _drift_record(a_star, c, pendulum=kind == 'pendulum') for kind in ('flat', 'pendulum')}


def test_kinematic_drift_of_the_slow_centre(drift_records):
    analysis = analyse_mmr(drift_records['flat'], S, scan_owner=False, scan_subresonances=False)
    kinematics, verdict = analysis.kinematics, analysis.verdict
    assert kinematics.centre_corr_rate >= 0.99
    assert kinematics.centre_slope == pytest.approx(1.0, abs=0.01)
    assert kinematics.centre_intercept_rad_yr == pytest.approx(LAMBDA_RATE, rel=0.01)
    share = analysis.extra_columns['kin_centre_ripple_share']
    assert 0.02 < share < 0.1  # 40 deg against ~1065 deg
    assert analysis.extra_columns['rec_rule'] == decide.RULE_K2
    assert (verdict.status, verdict.reason, verdict.libration_origin, verdict.origin_detail) == (
        Status.NON_RESONANT,
        Reason.KINEMATIC_DRIFT,
        LibrationOrigin.KINEMATIC,
        'varpi_term',
    )
    # author 2026-10-19 q7: the slow centre's word. It never passes a revolution in one direction (the lambda part
    # steps it back while p varpi_dot < 0.0292 deg/yr) but advances (0.0505 - 0.0292) deg/yr x 50 kyr = 1065 deg =
    # 2.96 rev net >= libration_circulation_net_rev: a circulation by G4b (W22 R-NET; the real 21028 has 1.81 rev: drift)
    assert analysis.geometry.centre.max_pass_rev < 1 and analysis.geometry.centre.net_rev == pytest.approx(2.96, abs=0.05)
    assert verdict.angle_status == AngleStatus.CIRCULATION and verdict.final_regime == ''
    assert verdict.checks['check_mmr_kinematic'] == CheckValue.FAIL
    strict = analyse_mmr(drift_records['flat'], replace(S, libration_circulation_net_rev=3.0), scan_owner=False, scan_subresonances=False)
    assert strict.verdict.angle_status == AngleStatus.DRIFT  # 2.96 < 3: a drift as before W22


def test_the_angle_word_setting(drift_records):
    settings = replace(S, mmr_kinematic_drift_word=False)
    verdict = analyse_mmr(drift_records['flat'], settings, scan_owner=False, scan_subresonances=False).verdict
    assert verdict.reason == Reason.KINEMATIC_DRIFT and verdict.angle_status == AngleStatus.EPISODIC


def test_a_pendulum_response_at_the_ripple_is_not_kinematic(drift_records):
    analysis = analyse_mmr(drift_records['pendulum'], S, scan_owner=False, scan_subresonances=False)
    assert analysis.response.force_r2_h1 >= S.mmr_force_r2_fail  # a answers sin sigma: the premise of K2 fails
    assert analysis.extra_columns['rec_rule'] != decide.RULE_K2
    assert analysis.verdict.reason not in (Reason.KINEMATIC_DRIFT, Reason.KINEMATIC_LIBRATION)


@pytest.mark.benchmark
def test_21028_is_a_kinematic_drift():
    example = 'mmr3__21028__7E-10M-6+0+0+9'
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    analysis = analyse_mmr(io.load_series(example), S)
    verdict = analysis.verdict
    assert (verdict.status, verdict.angle_status, verdict.reason) == (Status.NON_RESONANT, AngleStatus.DRIFT, Reason.KINEMATIC_DRIFT)
    # Q21 (F report, Codex): the slow sigma rate is 9 varpi_dot - 0.029 deg/yr, corr 0.997, slope 1.005
    assert analysis.kinematics.centre_corr_rate >= 0.99
    assert analysis.kinematics.centre_slope == pytest.approx(1.0, abs=0.05)
    assert math.degrees(analysis.kinematics.centre_intercept_rad_yr) == pytest.approx(-0.029, abs=0.002)


@pytest.mark.benchmark
def test_a_hilda_is_not_a_kinematic_drift():
    """153 3J-2: its centre follows the forced varpi (corr 0.98), but a answers sigma and its swing is the centre's
    whole range."""
    example = 'mmr2__153__3J-2+0-1'
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    analysis = analyse_mmr(io.load_series(example), S)
    assert analysis.verdict.status == Status.RESONANT
    assert analysis.extra_columns['kin_centre_ripple_share'] > S.mmr_kin_ripple_share_max


@pytest.mark.benchmark
def test_2897_is_no_k2_drift_while_its_multiplet_law_holds():
    """2897 35S-4 (review B #1): the slow centre runs 10.8 rev with 31 varpi (corr 1.000), the harmonic law fails in
    both halves (R^2 0.20), but the multiplet law reaches R^2 0.50 >= MMR_FORCE_R2_FAIL (0.4): a answers sigma under
    one law, so K2 does not fire and rule 5s decides (non_resonant/implausible_strength, as before CS7)."""
    example = 'mmr2__2897__35S-4+0-31'
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    analysis = analyse_mmr(io.load_series(example), S)
    verdict = analysis.verdict
    assert analysis.response.multiplet.r2 >= S.mmr_force_r2_fail
    assert analysis.extra_columns['rec_rule'] != decide.RULE_K2
    assert (verdict.status, verdict.reason) == (Status.NON_RESONANT, Reason.IMPLAUSIBLE_STRENGTH)
    entry = next(
        e for e in verdict.decision_path if e.name == '3 kinematic (slow centre): a does not answer sigma, force law R2 (multiplet)'
    )
    assert entry.value == pytest.approx(analysis.response.multiplet.r2) and entry.result is False


def test_k2_needs_every_force_law_to_fail(drift_records):
    """The flat synthetic record is K2. The same numbers with the multiplet law at R^2 0.99 (halves still below the bar) are not,
    and neither with the multiplet not measured: the decision and the ledger are one walk, the ledger's last force-law
    entry says why."""
    analysis = analyse_mmr(drift_records['flat'], S, scan_owner=False, scan_subresonances=False)
    response, kinematics, geometry = analysis.response, analysis.kinematics, analysis.geometry
    angle = decide.AngleDecision(AngleStatus.EPISODIC, decide.AngleSubtype.RECURRENT, 'G2', analysis.decisive_episode_id)
    assert max(response.force_r2_h1, response.force_r2_h2) < S.mmr_force_r2_fail
    assert decide.slow_centre_is_kinematic(angle, geometry, response, kinematics, S)
    held = replace(response, multiplet=replace(response.multiplet, r2=0.99))
    unmeasured = replace(response, multiplet=None)
    for changed, value in ((held, 0.99), (unmeasured, None)):
        assert not decide.slow_centre_is_kinematic(angle, geometry, changed, kinematics, S)
        ledger = Ledger.from_settings(S)
        assert not decide._slow_centre_route(angle, geometry, changed, kinematics, S, ledger)
        assert ledger.entries[-1].name.endswith('(multiplet)') and ledger.entries[-1].value == value


def test_k2_grades_the_centre_as_a_drift(drift_records):
    """Review B #6: under K2 the margin K1 reads the slow centre's correlation (the one K2 decided on) and B5 is graded
    (K2's verdict is a drift); the episode margins L1-L7 are not (the episodes are a ripple on the drift)."""
    analysis = analyse_mmr(drift_records['flat'], S, scan_owner=False, scan_subresonances=False)
    margins = analysis.verdict.margins
    assert margins['K1'].value == pytest.approx(analysis.kinematics.centre_corr_rate)
    assert 'B5' in margins and not {'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7'} & set(margins)


@pytest.mark.benchmark
def test_the_edge_rule_reads_the_real_last_episode():
    """145480 1N-4 at 1 Myr (review B #9): the real episode 0 ends at 680.9 kyr, the forced waves 1 and 2 come after it
    (682.0-683.6 and 730.0-734.6 kyr). Before round C the exit was read after wave 2 (758.1 kyr); now it is read after
    episode 0, before the second wave starts, and no forced wave keeps an edge mark."""
    example = 'mmr2__145480__1N-4+0+3__1Myr'
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    geometry = analyse_mmr(io.load_series(example), S).geometry
    (real,) = geometry.episodes
    waves = [run for run in geometry.runs if run.episode_id in geometry.forced_wave_ids]
    assert len(waves) == 2 and all(real.t_end_yr < wave.t_start_yr for wave in waves)
    assert real.t_end_yr < geometry.exit.time_yr < max(wave.t_start_yr for wave in waves)
    assert not any(wave.exit_observed or wave.entry_observed for wave in waves)
