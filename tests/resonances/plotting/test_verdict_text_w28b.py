"""W28 E19a: the words and figure marks of the B-series rules (cards g11-1, g03-4, g08-3, g03-5, g08-4, g07-5, g08-2,
g08-1, g07-1, g07-2, g07-3, g08-5, g08-6, g06-6, g11-2, g04-5).

Fast tests build real result objects on the 2:1 kinematic synthetic of test_verdict_text_roles (`ForcedWave`,
`DirectForceLaw`, `LedgerEntry`, `EnergyAnalysis`, geometry and episode builders) and set the columns a sentence reads
to numbers of the benchmark rows the cards name (2012 KL9, 6318, 2004 YG1, 523683, 2001 VC2, 2004 BG41, 59358,
2005 GQ107, 79653, 2003 TX9, 2003 EH16, 385266). Every expected string is worked out by hand from those numbers and
the settings (asserted first), not printed by the code. The benchmark tests read 59358 (panel (e)), 523683 and 82831.
"""

from __future__ import annotations

import math
from dataclasses import replace

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from resonances.plotting import mmr_combined as mc  # noqa: E402
from resonances.plotting import mmr_verdict_panel as vp  # noqa: E402
from resonances.plotting import verdict_text as vt  # noqa: E402
from resonances.resonance.analysis.ledger import LedgerEntry, Op  # noqa: E402
from resonances.resonance.analysis.mmr import response as R  # noqa: E402
from resonances.resonance.analysis.mmr.decide import O1_ONSET, SCALE_OUT_OF_RANGE  # noqa: E402
from resonances.resonance.analysis.mmr.models import DirectForceLaw, ForcedWave  # noqa: E402
from resonances.resonance.analysis.settings import settings_from_config  # noqa: E402
from resonances.resonance.analysis.verdict import CheckValue  # noqa: E402
from tests.resonances.analysis.builders import make, make_energy, make_episode, make_geometry, make_response  # noqa: E402
from tests.resonances.plotting.test_verdict_text_roles import _kinematic  # noqa: E402

S = settings_from_config()


def test_the_settings_the_expected_strings_assume():
    """The hand-worked strings below read these values (`.env.dist`)."""
    assert (S.mmr_force_r2_fail, S.mmr_force_r2_full, S.mmr_force_null_gain) == (0.4, 0.7, 0.2)
    assert (S.mmr_strength_check_ratio, S.mmr_direct_force_cutoff_cycles, S.mmr_a_response_min_amp_deg) == (10.0, 0.35, 45.0)
    assert (S.mmr_o1_coherence_r2_min, S.mmr_o1_phase_gap_max_deg, S.mmr_o1_own_law_r2_fraction) == (0.5, 90.0, 0.5)
    assert (S.mmr_identity_slope_tol, S.mmr_identity_corr_min, S.separatrix_vevf_fraction) == (0.2, 0.9, 0.666)
    assert (S.mmr_filter_orbit_factor, S.mmr_extend_force_windows, S.mmr_force_window_cycles) == (1.5, 2.0, 3.5)


def _facts(row=None, verdict=None, **analysis):
    """MMRFacts of the synthetic (`_kinematic`, 2J-1+0-1, Jupiter, p = −1) with its analysis fields, verdict fields
    and row columns replaced."""
    base, columns = _kinematic()
    v = replace(base.verdict, **(verdict or {}))
    a = replace(base, verdict=v, **analysis)
    return vt.MMRFacts(a, S, {**columns, **(row or {})}, a.episode_rows())


def _wave(episode_id, **fields):
    defaults = dict(n_tp=5, period_yr=214.0, identity_slope_ratio=float('nan'), identity_corr=float('nan'), force_line='')
    defaults.update(force_line_offset=float('nan'), wrong_sign=False, on_line=False)
    defaults.update(fields)
    return ForcedWave(episode_id=episode_id, **defaults)


# --------------------------------------------------------------------------------------------- B2: forced (strength)


def test_b2_a_5s_row_names_its_strength_marks_and_no_driver():
    """g11-1 A (2012 KL9): the two accepted runs of a 5s row are drawn as forced (strength); the text names the ground
    (the angle's own force cannot make that period) and says the driver is not identified."""
    geometry = make_geometry(forced_waves=(_wave(0, by_strength=True), _wave(2, by_strength=True)))
    f = _facts(geometry=geometry)
    assert vt.strength_mark_sentence(f) == (
        ' Its 2 accepted episodes are drawn as forced (strength): a period this angle’s force cannot make; the driver is ' 'not identified.'
    )
    one = _facts(geometry=make_geometry(forced_waves=(_wave(0, by_strength=True),)))
    assert vt.strength_mark_sentence(one).startswith(' Its accepted episode is drawn as forced (strength)')
    assert vt.strength_mark_sentence(_facts(geometry=make_geometry())) == ''


def test_b2_b6b_the_figure_labels_and_the_episode_lines_name_the_mark():
    """(a)/(l) label '#id forced (strength)' and '#id ripple' (dotted fill), '#id wave' for F6; the legend names all
    three; the numbers' episode lines print the route's words."""
    geometry = make_geometry(forced_waves=(_wave(0, by_strength=True), _wave(1, ripple=True), _wave(2, on_line=True)))
    assert mc.wave_labels(geometry) == {0: ' forced (strength)', 1: ' ripple', 2: ' wave'}
    runs = [make_episode(episode_id=i, tp_times_yr=np.array([0.0, 400.0, 800.0]) + 1000.0 * i) for i in range(3)]
    fig, ax = plt.subplots()
    mc.shade_runs(ax, runs, {0, 1, 2}, {}, lambda t: np.asarray(t), 1.0, mc.LAYOUT_V1, forced_ids=mc.wave_labels(geometry))
    texts = sorted(text.get_text() for text in ax.texts)
    assert texts == ['#0 forced (strength)', '#1 ripple', '#2 wave']
    assert any(patch.get_hatch() == mc.RIPPLE_HATCH for patch in ax.patches)
    plt.close(fig)
    legend = ' '.join(mc.LEGEND)
    assert 'forced (strength)' in legend and 'ripple (dotted)' in legend
    episodes = [
        {'episode_id': 0, 'accepted': True, 'forced_wave': 'strength', 't_start_yr': 8600.0, 't_end_yr': 9240.0},
        {'episode_id': 1, 'accepted': True, 'forced_wave': 'unconfirmed_ripple', 't_start_yr': 13468.0, 't_end_yr': 14894.0},
        {'episode_id': 2, 'accepted': True, 'forced_wave': 'lattice_line', 't_start_yr': 20000.0, 't_end_yr': 30000.0},
    ]
    lines = vp.explained_episode_lines(episodes)
    assert lines[0].group == 'EPISODES (3 accepted (1 forced (strength), 1 unconfirmed ripple, 1 forced wave), 0 rejected runs)'
    assert [line.value.split(':')[-1] for line in lines] == [' forced (strength)', ' unconfirmed ripple', ' forced wave']


# --------------------------------------------------------------------------------------------- B4: O1 in three parts


def test_b4_o1_prints_its_three_parts_with_the_numbers():
    """g03-4 A / g08-3 A: 6318 3J-1 (coherence 0.344, gap 2.17°, own law 0.6 of 0.635 = 0.945): not a foreign wave;
    a coherent, in-phase wave the own law explains 0.1/0.8 = 0.125 of: a foreign wave."""
    f = _facts()
    own = {'onset_coherence_r2': 0.344, 'onset_phase_gap_deg': 2.17, 'onset_own_law_r2': 0.6, 'onset_own_law_r2_episode': 0.635}
    assert vt.foreign_wave_text(f, own) == (
        'coherent at P: R² 0.344 < 0.5; phase gap 2.17° ≤ 90°; own law explains 0.945 ≥ 0.5 of the episode’s R² → not ' 'a foreign wave'
    )
    foreign = {'onset_coherence_r2': 0.69, 'onset_phase_gap_deg': 40.0, 'onset_own_law_r2': 0.1, 'onset_own_law_r2_episode': 0.8}
    assert vt.foreign_wave_text(f, foreign) == (
        'coherent at P: R² 0.69 ≥ 0.5; phase gap 40° ≤ 90°; own law explains 0.125 < 0.5 of the episode’s R² → a foreign wave'
    )
    assert vt.foreign_wave_text(f, {}).endswith('own law not measured → not a foreign wave')


def test_b4_the_5o_objection_no_longer_says_strong_a_wave():
    """g03-5 A: the 5o clause words O1 as 'before the episode a already oscillated with comparable strength', with the
    ratio as it came out (2013 VA14 60 kyr: 0.733 ≥ 0.5)."""
    entry = LedgerEntry(O1_ONSET, 0.733, Op.LT, 0.5, result=False)
    f = _facts(verdict={'decision_path': [entry]})
    text = vt._objection_clause(f)
    assert text == (
        'a measured test argues against the libration: before the episode a already oscillated with comparable '
        'strength (before/inside 0.733 ≥ 0.5)'
    )
    assert 'strong a-wave' not in text


# --------------------------------------------------------------------------------------------- B6b, B6c, B6a: the slow swing


def test_b6b_the_ripple_is_an_unconfirmed_auxiliary_interval():
    """g08-4 A (2004 YG1 20 kyr): the 3.11° ripple #0 of P 680 yr whose identity fails (slope/c 0.65) is set aside;
    step 0 reads [F6r, G5a] when the slow centre turned back once."""
    ripple = _wave(0, period_yr=680.17, identity_slope_ratio=0.65, identity_corr=0.794, ripple=True, half_amp_max_rad=math.radians(3.1075))
    f = _facts(geometry=make_geometry(forced_waves=(ripple,)))
    assert vt.ripple_clause(f) == (
        'the ripple #0 (P 680 yr, half-amplitude 3.11° < 45°: no force law or potential measurable; identity |slope/c − 1| '
        '0.35 > 0.2) is set aside as an unconfirmed ripple, an auxiliary interval'
    )
    assert vt.ripple_clause(_facts(geometry=make_geometry())) == ''


SLOW_SWING_523683 = {  # 523683 2U-9+0+7 (B6c measure_g5w.py; geo_slow_swing_*)
    'geo_slow_swing_t0_yr': 11929.0,
    'geo_slow_swing_t1_yr': 67710.0,
    'geo_slow_swing_amp_deg': 113.708,
    'geo_slow_swing_period_yr': 111562.0,
    'geo_slow_swing_identity_slope_ratio': 1.001862,
    'geo_slow_swing_identity_corr': 0.99998,
    'geo_slow_swing_a_ratio': 1.07537,
    'geo_slow_swing_period_ratio': 1.00411,
    'geo_slow_swing_t0_theory_yr': 84031.1,
    'geo_centre_range_rad': 3.969166,
    'rec_rule': 'G6',
    'flt_cutoff_binding': 'orbit',
    'ctx_P_orb_yr': 379.144,
    'flt_cutoff_yr': 568.716,
}


def test_b6c_g5w_reads_the_half_swing_in_the_well_and_the_orbital_period():
    """g07-5 A (523683 2U-9): step 0 is G5w with the slow identity, the a and period ratios, and 'the 379-yr cycles are
    the orbital period' (the cutoff 1.5 P_orb, B6a); the subtitle names G5w, not the tree's generic G6. 3.969 rad /
    2π = 0.63 rev; |1.001862 − 1| = 0.00186; 1/√10 = 0.32."""
    verdict = {'status': 'uncertain', 'angle_status': 'insufficient_time', 'angle_subtype': 'arc', 'rules_fired': ['slow_swing_in_well']}
    f = _facts(SLOW_SWING_523683, verdict)
    assert vt.slow_swing_decided(f)
    assert vt.slow_swing_text(f) == (
        'the slow centre turns back twice (11.9 kyr, 67.7 kyr, within 0.63 rev): the 114° half-swing between them is '
        'this resonance’s well: slow identity |slope/c − 1| 0.00186 ≤ 0.2, corr 1 ≥ 0.9; a swing 1.08 × the pendulum’s, '
        'period 112 kyr = 1 × T0_th F(A) (T0_th 84 kyr), both in [0.32, 3.16]'
    )
    assert vt.orbit_period_clause(f) == 'the 379 yr cycles are the orbital period (cutoff 569 yr = 1.5 P_orb)'
    assert vt.shape_rule_ids(f) == ('G5w',)
    assert vt.shown_rule(f) == 'G5w'
    assert vt._insufficient_text(f, 'arc').endswith('(cutoff 569 yr = 1.5 P_orb) → insufficient_time (arc)')
    assert 'turns back twice' in vt.mmr_angle_sentence(f)
    synodic = _facts({**SLOW_SWING_523683, 'flt_cutoff_binding': 'synodic'}, verdict)
    assert vt.orbit_period_clause(synodic) == ''


# --------------------------------------------------------------------------------------------- B5: G1f


def test_b5_g1f_says_which_floor_and_what_confirmed_the_cycle():
    """g08-2 B (2001 VC2 16E-17): the 105-yr cycle below the 150-yr floor, kept at T0_th / √10 = 111.75 / 3.162 =
    35.3 yr; step 0 lists G1f."""
    entry = LedgerEntry(
        "G1f floor of the angle's own averaged potential T0_th / sqrt(MMR_STRENGTH_CHECK_RATIO)", 35.34, Op.LT, 150.0, result=True
    )
    row = {
        'geo_subfloor_kept': 1.0,
        'geo_subfloor_period_yr': 105.026,
        'geo_subfloor_t0_theory_yr': 111.755,
        'geo_subfloor_floor_yr': 35.34,
    }
    f = _facts(row, {'decision_path': [entry]})
    assert vt.subfloor_clause(f) == (
        'a two-body cycle of 105 yr, below the 150 yr floor, read again at the floor 35.3 yr of its own averaged potential '
        '(T0_th 112 yr / √10) and kept: F by direct averaging, the centre and the force law (L4) confirm it'
    )
    assert 'G1f' in vt.shape_rule_ids(f)
    assert vt.subfloor_clause(_facts({**row, 'geo_subfloor_kept': 0.0})) == ''


# --------------------------------------------------------------------------------------------- B3: R0 over eligible windows


def test_b3_r0_counts_k_of_n_eligible_windows_on_the_visited_arc():
    """g08-1 A (2004 BG41 3J-1): 18 of 22 eligible windows agree (82% ≥ 67%), 60 − 22 = 38 swing less than 45°."""
    energy = make_energy(ve_vf_agree_fraction=18 / 22)
    row = {'sep_VE_VF_n_eligible': 22, 'sep_VE_VF_n_windows': 60, 'sep_VE_VF_arc_deg': 165.04}
    f = _facts(row, energy=energy)
    assert vt.window_count_text(f) == (
        '18 of 22 eligible windows (82% ≥ 67%); 38 of 60 windows swing less and are not counted; compared on the visited ' 'arc of 165°'
    )
    detail, need = vt._potential_detail(f)
    assert detail.startswith('V_E ≈ V_F in 18 of 22 eligible windows') and 'half-swing ≥ 45° or a revolution' in need
    assert vt.window_count_text(_facts({'sep_VE_VF_n_eligible': 0, 'sep_VE_VF_n_windows': 60}, energy=energy)) == ''


# --------------------------------------------------------------------------------------------- B8, B8b: the direct law, the lines


LAW_59358 = dict(  # 59358 8N-13 (B8 research; rsp_dforce_*)
    period_yr=4353.5, period_large_yr=8782.5, survived=True, cutoff_yr=3073.875, n_windows=4, r2=0.7548,
    scale_min=1.1374, scale_max=1.2857, scale_med=1.247, null_max=0.0174, taken=True,
)  # fmt: skip


def test_b8_the_l4_row_names_the_direct_law_and_its_numbers():
    """g07-2 A: L4 decided by the term's averaged force: 3073.875 yr = 0.35 × 8782.5 yr; R² 0.7548 ≥ 0.4 (full, ≥
    0.7); s 1.14–1.29 within [1/√10, √10] = [0.32, 3.16]; above the null by 0.7548 − 0.0174 = 0.737; the harmonic law's
    halves stay printed."""
    law = make(DirectForceLaw, **LAW_59358)
    base, _ = _kinematic()
    response = replace(base.response, force_law='direct_force', force_direct=law, force_r2_h1=0.163, force_r2_h2=0.168)
    f = _facts(response=response)
    assert vt.direct_force_text(f, law, 'R² 0.163/0.168') == (
        'averaged force of 2J-1+0-1 (Jupiter’s real mass) at 3.07 kyr = 0.35 P 8.78 kyr (the large turns): R² 0.755 ≥ '
        '0.4 (full), s 1.14–1.29 within [0.32, 3.16], above the shifted null by 0.737 ≥ 0.2, 4 windows; harmonic law R² '
        '0.163/0.168'
    )
    assert vt._margin_name(f, 'L4') == vt.DIRECT_FORCE_NAME
    parts = vt._direct_law_parts(f)
    assert [part.column for part in parts] == ['rsp_dforce_R2', 'rsp_dforce_scale_min', 'rsp_dforce_scale_max', 'rsp_dforce_null_gain']
    assert [part.text for part in parts] == ['0.755 ≥ 0.4', '1.14 ≥ 0.316', '1.29 ≤ 3.16', '0.737 ≥ 0.2']
    assert mc.direct_law_text(law, f.a) == "direct law 2J-1+0-1 (Jupiter's mass) at 3.07 kyr: R² 0.75, s 1.14–1.29"
    refused = replace(response, force_direct=replace(law, taken=False))
    assert vt._direct_law_parts(_facts(response=refused)) == ()


def test_b8_panel_e_redraws_the_law_the_decision_measured():
    """`response.direct_window_fits` (what panel (e) draws) is the comparison `judge_direct` made: on the B8 synthetic
    (a pendulum plus a 2.1-kyr line, s = 1 by construction) the smaller half median of its window R² is the law's R²."""
    from tests.resonances.analysis.test_direct_force_law import T, _judge, _pendulum, _profiles

    sigma, a, period = _pendulum()
    law = _judge(sigma, a, period, _profiles())
    cutoff = S.mmr_direct_force_cutoff_cycles * period
    windows = R.direct_windows(T, [(T[0], T[-1])], period, cutoff, S)
    observed, fits = R.direct_window_fits(T, a, np.mod(sigma, 2 * np.pi), windows, _profiles(), cutoff, S)
    assert len(observed) == len(T) and len(fits) == law.n_windows
    assert R.lower_half_median([w[0] for w in windows], np.array([fit[3] for fit in fits])) == pytest.approx(law.r2, abs=1e-12)
    assert min(fit[2] for fit in fits) == pytest.approx(law.scale_min, abs=1e-12)
    assert max(fit[2] for fit in fits) == pytest.approx(law.scale_max, abs=1e-12)


GQ107_LINES = {  # 2005 GQ107 40M-57 (B8b; rsp_line_*)
    'rsp_line_ids': '57 P_M = P_conj; 7M-10',
    'rsp_line_period_yr': 107.127,
    'rsp_line_share': 0.9326,
    'rsp_line_share_all': 0.9317,
    'rec_rule': '5a*',
}


def test_b8b_the_line_share_is_said_only_where_the_force_law_fails():
    """g07-3 A: GQ107's ȧ is mostly the 107-yr line 57 P_M (93 %); 59358's 2.14-kyr line 13 P_N carries 44 %, the
    lattice 83 %; nothing where L4 passes, at a kinematic leaf, or for the libration's own period."""
    failed = {'checks': {**_kinematic()[0].verdict.checks, 'check_mmr_a_response': CheckValue.FAIL}}
    f = _facts(GQ107_LINES, failed)
    assert vt.line_share_line(f) == 'ȧ’s oscillation is mostly the 107 yr line 57 P_M = P_conj (7M-10): 93% of its variance.'
    lines_59358 = {
        'rsp_line_ids': '13 P_N = P_conj; 3N-5; 5N-8', 'rsp_line_period_yr': 2142.17, 'rsp_line_share': 0.4401,
        'rsp_line_share_all': 0.8287, 'rec_rule': '5d/6',
    }  # fmt: skip
    assert vt.line_share_line(_facts(lines_59358, failed)) == (
        'ȧ’s oscillation is mostly the 2.14 kyr line 13 P_N = P_conj (3N-5; 5N-8): 44% of its variance, the lattice lines ' 'together 83%.'
    )
    passed = {'checks': {**_kinematic()[0].verdict.checks, 'check_mmr_a_response': CheckValue.PASS}}
    assert vt.line_share_line(_facts(GQ107_LINES, passed)) == ''
    assert vt.line_share_line(_facts({**GQ107_LINES, 'rsp_line_ids': 'P_lib'}, failed)) == ''
    assert 'Radial overlap' not in vt.line_share_line(f)  # E6's line is the orbit slot's, not repeated
    assert vt.line_share_line(_facts({**GQ107_LINES, 'rec_rule': '3'}, failed)) == ''  # a kinematic leaf: a carries no swing


# --------------------------------------------------------------------------------------------- B9: 5x, A-FW, EH16


def test_b9_5x_is_a_diagnostic_not_an_objection():
    """g08-6 A (79653): the order-of-magnitude scale 39.5 × μ e^q lies outside [0.1, 10]: the origin is not
    established (two-body synthetic: μ e^q)."""
    entry = LedgerEntry(f'{SCALE_OUT_OF_RANGE}: required / theory <= x', 39.4747, Op.LE, 10.0, result=False)
    f = _facts({'rec_rule': '5x'}, {'decision_path': [entry]})
    assert vt._scale_clause(f) == (
        'the strength is known only as the order-of-magnitude scale μ e^q, and the period needs 39.5 × μ e^q, outside '
        '[0.1, 10]: a diagnostic, not a measured objection, so the origin is not established'
    )
    assert vt.branch_clause(f) == vt._scale_clause(f)
    assert '5x' in vt.BRANCH_LEAVES


def test_b9_the_extension_serves_the_late_regime():
    """g08-5 B (2003 TX9): L4 has too few windows; the last episode #0 reaches the end (from 2.21 kyr, P 3.14 kyr), so
    extend gives it 2 force windows of 3.5 P; the failed potential stays measured."""
    late = make_episode(episode_id=0, tp_times_yr=np.linspace(2210.0, 17910.0, 11), period_med_yr=3140.0, coverage_end_gap_p=0.1)
    geometry = make_geometry(runs=[late], window=(0.0, 20000.0))
    response = make_response(a_response=CheckValue.NOT_MEASURED, windows=[])
    checks = {**_kinematic()[0].verdict.checks, 'check_mmr_potential': CheckValue.FAIL, 'check_mmr_identity': CheckValue.PASS}
    verdict = {'status': 'uncertain', 'rules_fired': ['extend_length_limited'], 'extend_to_years': 30000.0, 'checks': checks}
    f = _facts({'rec_rule': '5e'}, verdict, geometry=geometry, response=response)
    assert vt.extension_line(f) == (
        'The extension serves the late regime: 2 force windows of 3.5 P for #0 (from 2.21 kyr, P 3.14 kyr); the failed '
        'potential stays measured, a longer record does not remove it.'
    )
    both = {**verdict, 'checks': {**checks, 'check_mmr_identity': CheckValue.FAIL}}
    assert vt.extension_line(_facts({'rec_rule': '5e'}, both, geometry=geometry, response=response)).endswith(
        'the failed identity and potential stay measured, a longer record does not remove them.'
    )
    assert vt.extension_line(_facts({'rec_rule': '5n'}, verdict, geometry=geometry, response=response)) == ''
    assert vt.extension_line(_facts({'rec_rule': '5e'}, {**verdict, 'rules_fired': []}, geometry=geometry, response=response)) == ''


def test_b9_eh16_a_suspected_kinematic_swing_with_its_shares():
    """g06-6 A (2003 EH16: kin_swing_share 0.768 > kin_lambda_swing_share 0.457) on the synthetic's −ϖ term: a
    suspicion with the shares, never on a resolved status or where the share does not exceed the λ part."""
    checks = {**_kinematic()[0].verdict.checks, 'check_mmr_a_response': CheckValue.NOT_MEASURED}
    row = {'kin_swing_share': 0.768, 'kin_lambda_swing_share': 0.457}
    f = _facts(row, {'status': 'uncertain', 'checks': checks})
    assert vt.kinematic_suspicion_line(f).startswith(
        'Suspected kinematic: over the episode −ϖ spans 0.77 of σ’s swing, the λ part 0.46; the force law that would '
        'tell it from a resonance is not measured ('
    )
    assert vt.kinematic_suspicion_line(_facts({**row, 'kin_swing_share': 0.3}, {'status': 'uncertain', 'checks': checks})) == ''
    assert vt.kinematic_suspicion_line(_facts(row, {'status': 'transient', 'checks': checks})) == ''


# --------------------------------------------------------------------------------------------- B7, B1


def test_b7_the_slow_scale_cap_names_what_decides():
    """g07-4 A (385533): the cap slow_scale_provisional is a LIMIT at medium whose other reading says a record of
    5.25 P decides."""
    (row,) = vt.cap_rows(['slow_scale_provisional'], 'medium')
    assert row.is_limit and row.short_name == 'slow scale, own force untested'
    assert 'a record ≥ 5.25 P decides' in row.alternative


def test_b1_385266_a_forced_wave_from_its_driver_never_a_capture():
    """g04-5 A: the loops of 385266 are 'a forced wave from driver 3N-5+0+2'; without a driver role the old words."""
    f = _facts(verdict={'cause_role': 'driver', 'owner': '3N-5+0+2'})
    assert vt.forced_wave_source(f) == 'a forced wave from driver 3N-5+0+2'
    assert vt.forced_wave_source(_facts()) == 'a forced wave of another argument'
    assert 'captur' not in vt.forced_wave_source(f)


# --------------------------------------------------------------------------------------------- benchmark rows


def _bench(example_id):
    from resonances.resonance.analysis.mmr import io
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    series = io.load_series(example_id)
    return analyse_mmr(series, S), series


@pytest.mark.benchmark
def test_59358_panel_e_draws_the_direct_law():
    """g07-2 A: 59358 8N-13's panel (e) draws the observed ȧ at the law's cutoff and s × the averaged force per window,
    and prints 'direct law 8N-13+0+5'; the L4 row names the averaged force."""
    analysis, series = _bench('mmr2__59358__8N-13+0+5')
    row = {**analysis.numbers_row(), **analysis.summary_row()}
    text = vt.explain_mmr(analysis, S, row, analysis.episode_rows())
    assert any(check.name == vt.DIRECT_FORCE_NAME for check in text.checks)
    fig, ax = plt.subplots()
    frame = mc._Frame(analysis, series, settings=S)
    mc.draw_adot_panel(ax, frame, mc.LAYOUT_V1)
    styled = [line for line in ax.lines if line.get_linewidth() == mc.DIRECT_MODEL_STYLE['lw'] and line.get_color() == mc.MODEL]
    assert styled and any('direct law 8N-13+0+5' in t.get_text() for t in ax.texts)
    plt.close(fig)


@pytest.mark.benchmark
def test_523683_and_82831_words():
    """523683: decided at G5w with the orbital period; 82831 13M-24: its early episode is never called a capture."""
    analysis, _ = _bench('mmr2__523683__2U-9+0+7')
    text = vt.explain_mmr(analysis, S, {**analysis.numbers_row(), **analysis.summary_row()}, analysis.episode_rows())
    assert text.decided_at.startswith('Decided at G5w') and 'orbital period' in text.steps[0].outcome
    analysis, _ = _bench('mmr2__82831__13M-24+0+11')
    text = vt.explain_mmr(analysis, S, {**analysis.numbers_row(), **analysis.summary_row()}, analysis.episode_rows())
    assert not any('captur' in bullet.text for bullet in text.bullets)
