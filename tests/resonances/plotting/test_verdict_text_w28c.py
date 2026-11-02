"""W28 E19b: the words of the co-orbital, record-edge, subresonance and joint-capture rules and the C11 figure style
(cards g02-9 A, g02-5 B, g02-6 B, g01-7 A, g06-1 A, g06-2 A, g06-4 A, g05-8 B; WPs C3, C4, C6, C11, C14, A6a, A6b,
A2b).

Fast tests build real result objects (`SingleLeg`, `LegTurn`, `HalfLibration`, `EdgeTurn`, geometry and episode
builders) on the 2:1 kinematic synthetic of test_verdict_text_roles and set the columns a sentence reads to the
numbers of the benchmark rows the cards name (2023 OE5, 2018 KS, 1991 VG, 2022 BB2, 2008 EZ84, 2003 YN107,
2006 BZ147, 2022 AP1, 42946 43S-6, 12218 4M-7, 2003 UG22 40 kyr, 2003 EH16), as one.py printed them on 2026-11-01.
Every expected string is worked out by hand from those numbers and the settings (asserted first): the outcome of a
leg from |ln out/in| and the zone against their settings, a span in the unit of its end, three significant digits.
"""

from __future__ import annotations

import math
from dataclasses import replace

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

from resonances.plotting import mmr_combined as mc  # noqa: E402
from resonances.plotting import mmr_verdict_panel as vp  # noqa: E402
from resonances.plotting import verdict_text as vt  # noqa: E402
from resonances.plotting import verdict_vocab as vv  # noqa: E402
from resonances.plotting.style import LAYOUT_V1, REJECTED  # noqa: E402
from resonances.resonance.analysis.geometry import EdgeTurn, HalfLibration, LegTurn, RejectReason, SingleLeg  # noqa: E402
from resonances.resonance.analysis.mmr import decide  # noqa: E402
from resonances.resonance.analysis.settings import settings_from_config  # noqa: E402
from tests.resonances.analysis.builders import make_episode, make_geometry  # noqa: E402
from tests.resonances.plotting.test_verdict_text_roles import _kinematic  # noqa: E402

S = settings_from_config()


def test_the_settings_the_expected_strings_assume():
    """The hand-worked strings below read these values (`.env.dist`)."""
    assert (S.mmr_coorbital_turn_energy_tol, S.mmr_coorbital_turn_zone_max) == (0.34, 1.1)
    assert (S.mmr_force_r2_fail, S.mmr_coorbital_edge_turn_orbits) == (0.4, 3.0)
    assert S.mmr_coorbital_single_leg and S.mmr_single_half_libration and not S.mmr_coorbital_leg_force_gate


def _facts(row=None, verdict=None, episodes=None, **analysis):
    """MMRFacts of the synthetic (2J-1+0-1, Jupiter) with its analysis fields, verdict fields, row columns and
    episodes.csv rows replaced."""
    base, columns = _kinematic()
    v = replace(base.verdict, **(verdict or {}))
    a = replace(base, verdict=v, **analysis)
    return vt.MMRFacts(a, S, {**columns, **(row or {})}, a.episode_rows() if episodes is None else episodes)


def _leg(t0, t1, topology, entry, exit, edge='', force='pass', scale=float('nan'), energy='pass'):
    turns = [LegTurn(t, out_in=o, zone=z) if t is not None else None for t, o, z in (entry, exit)]
    return SingleLeg(t0, t1, edge, topology, 1.5, turns[0], turns[1], force=force, force_scale=scale, energy=energy)


# 2023 OE5 1E-1: the horseshoe leg 586.312-664.814 yr, out/in 0.836 / 1.2, zone 0.866 / 0.862, K/K_th 1.24
OE5 = _leg(586.312, 664.814, 'horseshoe', (586.312, 0.836, 0.866), (664.814, 1.2, 0.862), scale=1.24)
# 2018 KS: the passing leg 778.566-933.519 yr (0.971 / 1.02, zone 0.834 / 0.906), law not measured
KS = _leg(778.566, 933.519, 'passing', (778.566, 0.971, 0.834), (933.519, 1.02, 0.906), force='not_measured', energy='not_measured')
# 1991 VG: the horseshoe leg 552.411-612.913 yr whose turns scatter (out/in 0.605 / 1.59)
VG = _leg(552.411, 612.913, 'horseshoe', (552.411, 0.605, 0.643), (612.913, 1.59, 0.732), scale=1.0)


def _coorbital(leg, rule, verdict, row=None):
    return _facts(
        row={'rec_rule': rule, 'geo_centre_max_pass_rev': 21.9, **(row or {})},
        verdict=verdict,
        geometry=make_geometry(coorbital=True, single_legs=(leg,)),
    )


# --------------------------------------------------------------------------------------------- C6: one 1:1 leg


def test_c6_the_leg_outcomes_the_strings_assume():
    """By hand: |ln 0.836| = 0.179, |ln 1.2| = 0.182 ≤ 0.34 and the zones ≤ 1.1 → both turns reflect, law and energy
    pass: transient; KS is a passing leg: uncertain (passing); VG |ln 0.605| = 0.503, |ln 1.59| = 0.464 > 0.34: both
    turns are kicks: scatter."""
    assert abs(math.log(0.836)) < 0.34 and abs(math.log(1.2)) < 0.34
    assert abs(math.log(0.605)) > 0.34 and abs(math.log(1.59)) > 0.34
    assert decide.leg_outcome(OE5, S) == (decide.LEG_TRANSIENT, '')
    assert decide.leg_outcome(KS, S) == (decide.LEG_UNCERTAIN, 'passing')
    assert decide.leg_outcome(VG, S) == (decide.LEG_SCATTER, 'turn')


def test_c6_a_clean_conservative_leg_is_one_horseshoe_leg_with_force_and_energy_conserved():
    """g02-5 B / g02-6 B (2023 OE5): the leaf G4c says the leg, its law, energy and turns; step 0 the circulation it
    judged; the leaf step is G4c; the verdict line names one leg and no closed cycle."""
    f = _coorbital(OE5, 'G4c', {'status': 'transient', 'reason': 'dynamic_episode', 'angle_status': 'episodic'})
    clause = (
        'one horseshoe leg 586–665 yr, force and energy conserved: its local 1:1 law holds (K/K_theory 1.24), its energy '
        'integral passes and both turns reflect (out/in 0.836 and 1.2; |ln out/in| ≤ 0.34); half a cycle, no closed one'
    )
    assert vt.leg_clause(f) == clause
    assert vt.mmr_status_sentence(f) == f'Transient: {clause}. Transient, not resonant, because one leg covers part of the record only.'
    assert (
        vt.verdict_line(f)
        == f'TRANSIENT: one leg in a co-orbital well of {f.v.commensurability or f.a.resonance.key}, held by the planet’s force; no closed cycle.'
    )
    steps = vt.mmr_steps(f)
    assert [step.number for step in steps] == ['0', 'G4c']
    assert steps[0].ids == ('G4',) and steps[0].outcome.endswith('→ a circulation judged on its best single leg → episodic')
    assert steps[1].ids == ('G4c',) and steps[1].outcome == f'{clause} → transient'
    assert vt.mmr_angle_sentence(f) == 'σ circulates: its centre passes 21.9 revolutions in one direction.'


def test_c6_a_passing_leg_is_uncertain_and_says_why():
    """g02-4 A (2018 KS): reason one_coorbital_leg in words, not 'one coorbital leg.'."""
    f = _coorbital(KS, 'G4c', {'status': 'uncertain', 'reason': 'one_coorbital_leg', 'angle_status': 'episodic'})
    assert vt.mmr_status_sentence(f) == (
        'Uncertain: one passing leg 779–934 yr between passages: it laps past the planet (it crosses both the planet and '
        'L3), not one swing of a co-orbital well, so no capture is shown and none excluded.'
    )
    assert vt.verdict_line(f) == 'UNCERTAIN: one leg near the planet does not show a capture, nor exclude one.'
    # 2022 BB2: |ln 0.64| = 0.446 > 0.34 (a kick), |ln 1.31| = 0.270 (a reflection); law and energy pass
    kick = _leg(527.211, 584.662, 'tadpole', (527.211, 0.64, 0.202), (584.662, 1.31, 0.276), scale=1.0)
    f = _coorbital(kick, 'G4c', {'status': 'uncertain', 'reason': 'one_coorbital_leg', 'angle_status': 'episodic'})
    assert vt.leg_clause(f) == (
        'one tadpole leg 527–585 yr between passages: a turn is a kick, not a reflection (out/in 0.64 and 1.31; a '
        'reflection keeps |ln out/in| ≤ 0.34), so no capture is shown and none excluded'
    )


def test_c6_a_scattering_leg_is_a_near_miss_in_the_circulation_leaf_and_the_cap():
    """g01-8 A (1991 VG): the circulation leaf names the leg whose turns are kicks; the near-episode cap says so."""
    f = _coorbital(VG, 'C', {'status': 'non_resonant', 'reason': 'circulation', 'angle_status': 'circulation'})
    leg = 'its best 1:1 leg 552–613 yr turns by kicks (out/in 0.605 and 1.59), not reflections'
    steps = vt.mmr_steps(f)
    assert steps[-1].number == 'C' and steps[-1].outcome.endswith(f'; {leg}')
    row = vt._cap_detail(f, vt.cap_rows(['near_episode'], 'medium')[0])
    assert row.name == 'a 1:1 leg almost made a capture, but both its turns are kicks, not reflections'
    assert row.short == 'leg 552–613 yr, out/in 0.605 and 1.59'


def test_c6_an_edge_leg_and_an_edge_half_swing_ask_for_the_rest_of_the_cycle():
    """q-4 A (a clean leg the record edge cuts) and q-5 i (42946 43S-6: one half-swing 28476.1-32285.5 yr to the end,
    R² 0.716 / 0.621 ≥ 0.4, restoring): baseline_needed with the swing, then the next check."""
    edge = _leg(933.519, 999.971, 'qs', (933.519, 1.02, 0.916), (None, None, None), edge='end')
    assert decide.leg_outcome(edge, S) == (decide.LEG_EXTEND, '')
    f = _coorbital(edge, 'G4c', {'status': 'uncertain', 'reason': 'baseline_needed', 'angle_status': 'insufficient_time'})
    assert vt.leg_clause(f) == (
        'one clean quasi-satellite leg 934–1000 yr cut by the record’s end: its turn at 934 yr reflects (out/in 1.02); '
        'the rest of the cycle lies beyond the end'
    )
    half = HalfLibration(28476.1, 32285.5, 'end', 1.0, 1.2, r2_h1=0.716, r2_h2=0.621, restoring=True)
    f = _facts(
        row={'rec_rule': 'G4h', 'geo_centre_max_pass_rev': 20.6},
        verdict={'status': 'uncertain', 'reason': 'baseline_needed', 'angle_status': 'insufficient_time'},
        geometry=make_geometry(half_librations=(half,)),
    )
    clause = (
        'one half-swing 28.5–32.3 kyr cut by the record’s end follows the resonance’s one-harmonic law (R² 0.716/0.621 ≥ '
        '0.4) and its force turns σ back at both turns; the rest of the cycle lies beyond the end'
    )
    assert vt.half_libration_clause(f) == clause
    assert vt.mmr_status_sentence(f).startswith(f'Uncertain: {clause}; ')
    assert vt.verdict_line(f) == 'UNCERTAIN: one half-swing of the resonance runs into the record’s edge.'
    assert [step.number for step in vt.mmr_steps(f)] == ['0', 'G4h']
    f = _facts(
        row={'rec_rule': 'G4h', 'geo_centre_max_pass_rev': 20.6},
        verdict={'status': 'uncertain', 'reason': 'one_half_libration', 'angle_status': 'episodic'},
        geometry=make_geometry(half_librations=(replace(half, edge=''),)),
    )
    assert vt.half_libration_clause(f).endswith('half a cycle, no closed one, so no libration is shown and none excluded')


# --------------------------------------------------------------------------------------------- C14: extend_time


def test_c14_the_flag_extend_time_says_where_and_why_with_the_next_check():
    """g01-7 A (2022 BB2: end slowdown and an end leg; 2008 EZ84: one reflection at the start; 42946: action extend,
    the action sentence names the next check)."""
    bb2 = _facts(
        row={'geo_extend_time_from': 'end:slowdown;leg:end'},
        verdict={
            'extend_direction': 'forward',
            'extend_to_years': 2000.0,
            'action': 'review',
            'final_regime': 'arc',
            'final_regime_start_yr': 361.0,
        },
    )
    assert vt.extend_flag_sentence(bb2) == (
        'Extend: a longer integration forward (next check: 2 kyr) would show how the record ends (σ_f slows to an arc '
        'after circulating; a clean 1:1 leg is cut by the end).'
    )
    assert vt._end_clause(bb2) == '; from 361 yr to the end σ_f slows to an arc (less than one revolution)'
    ez84 = _facts(row={'geo_extend_time_from': 'start:reflection'}, verdict={'extend_direction': 'backward', 'extend_to_years': 2000.0})
    assert vt.extend_flag_sentence(ez84) == (
        'Extend: a longer integration backward (next check: 2 kyr) would show how the record starts (σ turns back once at the start).'
    )
    extend = _facts(
        row={'geo_extend_time_from': 'half:end'}, verdict={'extend_direction': 'forward', 'extend_to_years': 1e5, 'action': 'extend'}
    )
    assert vt.extend_flag_sentence(extend) == (
        'Extend: a longer integration forward would show how the record ends (a half-swing that follows the law is cut by the end).'
    )
    assert vt.extend_flag_sentence(_facts()) == ''


# --------------------------------------------------------------------------------------------- C3, C4: 1:1 geometry


def test_c3_the_profile_floor_names_the_loop_and_both_floors():
    """g02-1 A (2003 YN107): loop 118.731 yr through two planet turns, floor 59.365 yr (0.5 × the loop), tadpole floor
    111.046 yr; only when rule coorbital_floor_profile acted."""
    episodes = [
        {
            'episode_id': 0,
            'accepted': True,
            't_start_yr': 95.55,
            't_end_yr': 147.0,
            'cpl_floor_source': 'profile_loop',
            'cpl_floor_yr': 59.365479,
            'cpl_loop_period_yr': 118.730958,
            'cpl_turn_sites': 'planet;planet',
        },
    ]
    row = {'flt_period_floor_yr': 111.046389}
    f = _facts(row=row, episodes=episodes, verdict={'rules_fired': ['coorbital_floor_profile']}, decisive_episode_id=0)
    assert vt.profile_floor_clause(f) == (
        '1:1 floor of #0: its turns reflect on the planet’s top of the averaged potential; the loop through them takes '
        '119 yr, so its own floor is 59.4 yr (the tadpole floor 111 yr)'
    )
    assert vt.profile_floor_clause(_facts(row=row, episodes=episodes, verdict={'rules_fired': []})) == ''


def test_c4_the_edge_turn_is_read_on_the_averaged_angle_and_only_closes_the_cycle():
    """g02-2 A (2006 BZ147): the exit turn at 960.47 yr, planet top, out/in 0.895, a crosses a_P."""
    turn = EdgeTurn('exit', 0, 960.47, 0.0, -1, 3.1, site='planet', out_in=0.895, a_crosses=True, outcome='closes')
    assert turn.added
    f = _facts(verdict={'rules_fired': ['coorbital_edge_turn']}, geometry=make_geometry(coorbital=True, edge_turns=[turn]))
    assert vt.edge_turn_clause(f) == (
        'the turn at 960 yr lies in the filter’s edge trim: read on σ averaged over 3 orbits, a reflection on the '
        'planet’s top (out/in 0.895, a crosses a_P); it only closes the cycle'
    )


# --------------------------------------------------------------------------------------------- C11: not counted


AP1_EPISODES = [
    {
        'episode_id': 0,
        'accepted': True,
        'eligible': False,
        'ineligible_reason': 'below_period_floor',
        'reject_reason': '',
        't_start_yr': 262.805,
        't_end_yr': 299.356,
        'n_tp': 2,
        'period_med_yr': 73.998,
        'half_amp_med_deg': 17.2,
        'half_amp_max_deg': 17.2,
    },
    {
        'episode_id': 1,
        'accepted': False,
        'eligible': False,
        'ineligible_reason': '',
        'reject_reason': 'over_the_top',
        't_start_yr': 413.0,
        't_end_yr': 413.0,
        'n_tp': 1,
    },
    {
        'episode_id': 2,
        'accepted': False,
        'eligible': False,
        'ineligible_reason': '',
        'reject_reason': 'over_the_top',
        't_start_yr': 915.6,
        't_end_yr': 915.6,
        'n_tp': 1,
    },
]


def test_c11_a_run_below_the_floor_is_not_counted_in_the_words_and_the_episode_list():
    """g02-9 A (2022 AP1): the accepted run #0 of P 74.0 yr against the Venus tadpole floor 75.67 yr is not counted:
    the sentence says so, step 0 names it, the episode list counts 1 accepted, 0 counted and greys its line."""
    f = _facts(row={'flt_period_floor_yr': 75.6716, 'geo_centre_max_pass_rev': 5.8}, episodes=AP1_EPISODES, geometry=None)
    assert (
        vt.not_counted_clause(f) == 'Not counted: #0 263–299 yr, P 74 yr < its floor 75.7 yr (faster than the angle’s shortest libration).'
    )
    assert vt.not_counted_clause(f, short=True) == '#0 P 74 yr < floor 75.7 yr: not counted'
    assert f.episodes.counted == [] and len(f.episodes.accepted) == 1
    lines = vp.explained_episode_lines(AP1_EPISODES)
    assert lines[0].group == 'EPISODES (1 accepted, 0 counted, 2 rejected runs)'
    assert lines[0].value.endswith(': not counted: below the period floor') and lines[0].style == 'grey'
    assert vp._episode_text(AP1_EPISODES[0]).endswith(' not counted: below the period floor')


def test_c11_the_figure_draws_an_uncounted_run_in_the_rejected_style_with_its_tag():
    """g02-9 A: on (a) the run is a grey strip at the bottom (the rejected style) with '#0 < floor' in grey above the
    panel, and no episode fill; a counted run keeps its coloured fill and bold '#1'."""
    floor_run = make_episode(2, episode_id=0, tp_times_yr=[262.8, 299.4], period_med_yr=74.0)
    counted = make_episode(3, episode_id=1, tp_times_yr=[500.0, 560.0, 620.0], period_med_yr=120.0)
    geometry = make_geometry(coorbital=True, runs=[floor_run, counted])
    geometry.eligibility = {0: RejectReason.BELOW_PERIOD_FLOOR, 1: RejectReason.NONE}
    tags = mc.uncounted_tags(geometry)
    assert tags == {0: ' < floor'}
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1000)
    mc.shade_runs(ax, geometry.runs, {0, 1}, {0: 'C0', 1: 'C1'}, lambda t: t, 1.0, LAYOUT_V1, uncounted=tags)
    spans = [(p.get_x(), p.get_x() + p.get_width(), p.get_facecolor()) for p in ax.patches]
    grey = [span for span in spans if span[2] == to_rgba(REJECTED, mc.REJECTED_ALPHA)]
    assert len(grey) == 1 and grey[0][:2] == pytest.approx((262.8, 299.4))
    strip = next(p for p in ax.patches if p.get_facecolor() == to_rgba(REJECTED, mc.REJECTED_ALPHA))
    assert strip.get_y() + strip.get_height() == pytest.approx(mc.REJECTED_STRIP)  # axes-fraction height of the strip
    texts = {t.get_text(): t for t in ax.texts}
    assert set(texts) == {'#0 < floor', '#1'}
    assert texts['#0 < floor'].get_color() == REJECTED and texts['#0 < floor'].get_fontweight() != 'bold'
    assert texts['#1'].get_fontweight() == 'bold'
    plt.close(fig)


# --------------------------------------------------------------------------------------------- A6a, A6b: multiplet


def test_a6a_the_12218_gap_is_where_the_composite_term_cancels():
    """g06-2 A (12218 4M-7: episodes 1.27-48.7 and 76.2-99.8 kyr, |P|/max|P| 8.84e-5 in the gap, 0.142 / 0.195 at its
    edges); here on the synthetic's planet, Jupiter. Not said when the gap's minimum is not below its edges."""
    episodes = [
        {'episode_id': 0, 'accepted': True, 't_start_yr': 1273.0, 't_end_yr': 48727.5, 'resonant_argument': '2J-1+1-2'},
        {'episode_id': 1, 'accepted': True, 't_start_yr': 76193.0, 't_end_yr': 99849.0, 'resonant_argument': ''},
    ]
    row = {'sub_th_P_min_gap': 8.83793e-05, 'sub_th_P_edges': '1.273:0.412;48.73:0.142;76.19:0.195;99.85:1', 'sub_choice': 'theory'}
    verdict = {'resonant_argument': '2J-1+1-2', 'commensurability': '2J-1'}
    f = _facts(row=row, verdict=verdict, episodes=episodes, decisive_episode_id=0)
    assert vt.gap_cause_line(f) == (
        'The gap 48.7–76.2 kyr between the episodes is where the composite term cancels: |P|/max |P| falls to 8.84e-5 '
        'there (0.142 and 0.195 at its edges) as ϖ_J − ϖ passes its cancellation phase.'
    )
    assert vt.gap_cause_line(_facts(row={**row, 'sub_th_P_min_gap': 0.5}, verdict=verdict, episodes=episodes)) == ''
    line = vt.argument_line(f)
    assert '(chosen by the multiplet theory: its centre π − arg P + jβ is the flattest over the episode)' in line
    assert line.endswith(' Other episodes: #1 76.2–99.8 kyr chose no member in its own window.')


def test_a6b_a_composite_argument_names_no_member():
    f = _facts(verdict={'resonant_argument': 'composite', 'commensurability': '2J-1'})
    assert vt.argument_line(f) == (
        'Resonant argument: composite, one resonance 2J-1 with this angle: no single member of the multiplet stands; σ '
        'follows the moving centre of the composite term P(β).'
    )


def test_a6a_the_cheap_list_frames_ug22_and_t8_is_a_diagnostic():
    """g06-4 A (2003 UG22 40 kyr: +5+2 and +6+1 flatter, W/W0 0.42 / 0.45, terms 7.6e-9 / 1.8e-10 of the strongest);
    a member flatter than the angle but strong enough (ok) or wider (W/W0 ≥ 1) is no frame; 2003 EH16's T8 scan."""
    cheap = '10E-17+5+2:0.42:7.6e-09:E_strength;10E-17+6+1:0.45:1.8e-10:E_strength;10E-17+4+3:0.51:3.2e-07:E_strength;X:0.3:0.5:ok;Y:1.2:1e-9:E_strength'
    f = _facts(row={'sub_cheap_list': cheap}, subresonances=None)
    assert vt.frame_line(f) == (
        'Members 10E-17+5+2 and 10E-17+6+1 give a flatter centre (W/W0 0.42, 0.45), but their terms are 7.6e-9 and '
        '1.8e-10 of the strongest (test E′): a frame, not a force; the centre moves with the body’s own e, i, ω.'
    )
    row = {'sub_diag_triggers': 'T8', 'sub_diag_best': '1M-3J-1-1+0+4', 'sub_diag_would_accept': '', 'sub_diag_would_check': 'not_measured'}
    assert vt.diagnostic_scan_line(_facts(row=row)) == (
        'A diagnostic subresonance scan (T8) ran: its flattest member is 1M-3J-1-1+0+4, check not measured; without a '
        'measured force it changes nothing.'
    )
    assert vt.diagnostic_scan_line(_facts(row={**row, 'sub_diag_triggers': ''})) == ''


# --------------------------------------------------------------------------------------------- A2b: joint capture


def test_a2b_both_resonances_at_once_in_the_role_the_cause_and_the_physics_row():
    """g05-8 B: origin joint, the owner keeps its row; x closes 18 cycles (period 1200 yr, secular offset 0.12)."""
    row = {'own_second_capture': 'confirmed', 'own_x_cycles': 18, 'own_x_cycle_period_yr': 1200.0, 'own_x_secular_offset': 0.12}
    verdict = {'libration_origin': 'joint', 'cause_role': 'joint', 'owner': '5J+6S-2+0+0-9', 'cause_term': ''}
    f = _facts(row=row, verdict=verdict)
    assert vt.cause_line(f.v, f) == ('joint', 'also in: 5J+6S-2+0+0-9 (both resonances at once)')
    assert vt.subtitle_role(f) == 'also in 5J+6S-2+0+0-9'
    assert vt.joint_line(f) == (
        'Both resonances at once: x = Ψ/u: x closes 18 cycles of its own well (period 1.2 kyr, off every secular period '
        'by 0.12) while 5J+6S-2+0+0-9 holds φ; σ librates in this resonance too.'
    )
    one = _facts(row={**row, 'own_second_capture': 'one_cycle', 'own_x_cycles': 1}, verdict=verdict)
    assert vt.joint_line(one).endswith(': not decided yet; a longer record decides.')
    assert 'x closes 1 cycle of its own well' in vt.joint_line(one)
    assert vt.CAPTURE_WORDS['confirmed'] == 'both resonances at once'
    assert vt.joint_line(_facts(row=row)) == ''  # origin not joint
    assert vv.rule_name('2j', leaf=True) and 'unmapped' not in vv.decided('2j') and 'unmapped' not in vv.decided('K3s')
