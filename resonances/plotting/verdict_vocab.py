"""The words of the explained right column (`verdict_text`): a registry of text, no logic.

Here, not in `verdict_text.py`, so that a work package that adds a cap, a reason or a rule adds its words without
touching the sentence code (a cap needs its `CapText`: `tests/resonances/plotting/test_w20_cs11_physics.py`):

- `RULE_NAMES`: every id the column can print (G rules, the leaves of the status tree, the margins and checks, the
  secular/ZLK margins `SZ.*`, the caps `cap:*`) → its short name, meaning first; `named` prints 'short name [id]',
  `decided` the subtitle 'decided: circulation (C)' (author 2026-10-31, card g10-3 C). A prefix only where an id
  collides: the secular/ZLK margins N, S, F, P, G are `SZ.N` … (MMR has N, S, F too), a cap is `cap:<key>` (several
  cap keys are also reasons or flags). The ids, the ledger entries and the CSV columns stay the code's; only the
  printed text changes. The table id → name → meaning is in docs/libration-analysis.md (#names).
- `CAPS` (`CapText`): how a cap is worded, its other reading and what decides it;
- the words of reasons, flags, ledger entries, margins and regimes (`*_WORDS`, `*_NAMES`, `REGIME_PHRASES`);
- `snake_words`: the fallback words of an identifier no table names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict

# The identifiers of the analysis (reasons, flags, CSV columns: centre_moves, geo_centre_*) keep their British
# spelling; the printed text is American (the author, 2026-11-05)
AMERICAN_WORDS = {'centre': 'center', 'centres': 'centers', 'neighbour': 'neighbor', 'neighbours': 'neighbors'}
AMERICAN_WORDS |= {'catalogue': 'catalog', 'behaviour': 'behavior', 'analysed': 'analyzed'}
_BRITISH = re.compile(r'\b(' + '|'.join(AMERICAN_WORDS) + r')\b')


def snake_words(name) -> str:
    """An identifier as words: 'centre_moves' -> 'center moves'."""
    return _BRITISH.sub(lambda m: AMERICAN_WORDS[m.group(0)], str(name).replace('_', ' '))


# --------------------------------------------------------------------------------------------- rule and check names

# id -> short name (meaning first, ≤ NAME_MAX characters, no two alike: tested). Ids of G rules, of the status tree's
# leaves (`rec_rule`; a path such as 5d/6 is named part by part, `rule_name`), of the margins and checks, of the
# secular/ZLK margins (SZ.) and of the caps (cap:, filled from `CAP_NAMES` below)
RULE_NAMES: Dict[str, str] = {
    # angle word (geometry.py, mmr/decide.py decide_angle)
    'G0': 'record unreadable',
    'G1': 'jump of a',
    'G1c': 'circulates after the libration',
    'G1e': 'libration before the jump',
    'G1l': 'capture after the jump',
    'G1q': 'quasi-satellite well',
    'G1f': 'cycle below the floor',
    'G2': 'librating episode',
    'G2u': 'episodes joined',
    'G2a': 'step of the action',
    'G2c': 'slow-center libration',
    'G3': 'stalls and slips at the separatrix',
    'G4': 'center makes a revolution',
    'G4b': 'net revolutions of the center',
    'G4c': 'single co-orbital leg',  # W28 C6: a 1:1 circulation judged on its best single leg
    'G4h': 'one half-swing between passages',  # W28 C6, q-5 i: not 1:1
    'G5': 'drifting center',
    'G5a': 'one slow arc',
    'G5w': 'slow swing in the well',
    'G6': 'too few cycles',
    # the leaves of the status tree (mmr/decide.py decide_status, the column rec_rule)
    'C': 'circulation',
    '1': 'a multiplet member librates',
    '2': 'another resonance drives a',
    '3': 'kinematic libration',
    '3k2': 'kinematic drift of the center',
    '3b5': 'slower libration possible',
    '4': 'unexplained drift',
    '5a': 'force law fails',
    '5a*': 'force law fails, no potential',
    '5b': 'force and potential disagree',
    '5c': 'potential supports',
    '5d': 'force measured',
    '5e': 'potential fails',
    '5f': 'force fit contaminated',
    '5g': 'identity fails',
    '5n': 'too few independent cycles',
    '5o': 'onset or strength objects',
    '5x': 'scale out of range',
    '5s': 'resonance too weak',
    '5s*': 'strength open (small divisors)',
    '5u': 'shape only',
    '6': 'judged by behavior',
    '6u': 'union covers the record',
    '2j': 'both resonances? one cycle of x',  # W28 A2b: 'both resonances at once?' with one cycle of x -> extend
    'K3s': 'slow scale is a secular cycle',  # W28 B10a: a circulation whose slow scale is a secular cycle
    # margins and checks (mmr/decide.py MARGIN_CLASSES, the tree's checks)
    'L1': 'survives the filter',
    'L2': 'away from the separatrix',
    'L3': 'σ̇ follows a − a*',
    'L4': 'a obeys the force law',
    'L5': 'one pendulum',
    'L6': 'fixed center',
    'L7': 'covers the record',
    'L8': 'resolved by step and filter',
    'L8a': 'step resolves σ',
    'L8b': 'filter passes the cycle',
    'L8c': 'filter guard compared',
    'B1': 'a carries the swing',
    'O1': 'no earlier a-wave',
    'R': 'witness episode',
    'R0': 'energy fits the potential',
    'Z': 'separatrix zone measured',
    'S': 'no member holds the center',
    'W': 'no other resonance drives a',
    'F': 'strength plausible',
    'N': 'not short of an episode',
    'F6': 'forced wave of another angle',
    'F6r': 'unconfirmed ripple',
    'F7': 'owner covers the episode',
    'C1': 'center passes whole revolutions',
    'C2': 'no reversals at any cutoff',
    'C3': 'steady rate',
    'C4': 'too fast for a pendulum',
    'C5': 'far from resonance',
    'K1': 'slow σ follows pϖ',
    'K1e': 'episode σ follows pϖ, a flat',
    'K2': 'slow center follows ϖ',
    'B5': 'slower libration excluded',
    'U': 'near its convention',  # a ledger comparison in the thin band of a convention (U:<entry>)
    # secular and Lidov-Kozai margins (secular/decide.py); SZ. tells them from the MMR margins N, S, F
    'SZ.N': 'enough reversals',
    'SZ.S': 'centers agree',
    'SZ.F': 'slower than the cutoff',
    'SZ.P': 'circulates at every smoothing',
    'SZ.G': 'free-ω gate',
}
# A leaf that is the outcome of a check or rule of the same id: the subtitle names the outcome ('decided: no resonant
# potential (R0)'), the check row the check ('energy fits the potential [R0]'). C5 decides non_resonant or invalid; G6
# is also the status leaf of every insufficient_time angle (rec_rule G6, `decide.decide_status`), whose word the angle
# rule set (column geo_angle_rule: G5a, G5w or G6, named by `ANGLE_SUBTYPE_NAMES` below)
LEAF_NAMES: Dict[str, str] = {
    'R0': 'no resonant potential',
    'C5': 'aliased circulation',
    'G6': 'too short to decide',
}
# angle rule -> {angle subtype: its name}, where one angle rule sets several words (`decide._g4_to_g6`: G6 few_cycles,
# arc or stall); a subtype not listed takes the rule's own name (G6 few_cycles: 'too few cycles')
ANGLE_SUBTYPE_NAMES: Dict[str, Dict[str, str]] = {
    'G6': {'arc': 'arc without turning points', 'stall': 'stall without turning points'},
}
# cap key -> short name; printed with the id cap:<key>
CAP_NAMES: Dict[str, str] = {
    'three_tp_episode': 'one closed cycle',
    'exit_suspected': 'exit unresolved',
    'entry_suspected': 'entry unresolved',
    'transient_unexplained': 'transition unexplained',
    'one_return_cycle': 'one phase-return cycle',
    'force_law_disagrees': 'force law against potential',
    'slow_libration_not_excluded': 'longer libration possible',
    'planets_missing': 'planets not saved',
    'owner_search_skipped': 'owner search off',
    'catalogue_longitudes': 'approximate planet longitudes',
    'nodal_subresonance_candidate': 'nodal member possible',
    'subresonance_tie': 'member chosen by order',
    'subresonance_t6_only': 'member scan on theory alone',
    'subresonance_theory_unconfirmed': 'member theory unconfirmed',
    'subresonance_undetermined': 'members indistinguishable',
    'jitter_low_e': 'pϖ jitter at low e',
    'all_physics_not_measured': 'own force unmeasured',
    'zone_not_measured': 'zone not measured',
    'l2_without_zone': 'separatrix distance unverified',
    'onset_failed': 'earlier a-wave',
    'no_full_physics_support': 'no full physical support',
    'force_law_theory_only': 'force law from theory only',
    'slow_scale_provisional': 'slow scale, own force untested',
    'episodes_in_separatrix_zone': 'episodes in the separatrix zone',
    'near_separatrix_without_zone': 'separatrix pattern without zone',
    'geometry_only': 'geometry only',
    'strength_scale_only': 'strength scale only',
    'implausible_strength': 'negative from strength only',
    'episode_before_jump': 'episodes before the jump',
    'episode_force_law_mixed': 'force law mixed over episodes',
    'near_episode': 'almost an episode',
    'short_period_aliased': 'short periods folded',
    'second_capture_unresolved': 'both resonances at once? open',  # card g05-5 A (W28 A3b)
    'one_cycle_energy_not_conserved': 'one-cycle energy fails',
    'action_step_provisional': 'action step provisional',
    'one_coorbital_leg': 'one co-orbital leg',  # W28 C6, leaf G4c
    'record_ends_unresolved': 'record ends in an arc',  # W28 C14, rule end_arc_extend
}
CAP_PREFIX = 'cap:'
RULE_NAMES.update({f'{CAP_PREFIX}{cap}': name for cap, name in CAP_NAMES.items()})
UNMAPPED = 'unmapped check'  # an id without a name is printed as such, never guessed (Codex, card g10-3)
LONG_PERIOD_PREFIX = 'SZ.'


def long_period_id(key: str) -> str:
    """The printed id of a secular/ZLK margin: 'N' -> 'SZ.N'."""
    return f'{LONG_PERIOD_PREFIX}{key}'


def rule_name(rule_id: str, leaf: bool = False, subtype: str = '') -> str:
    """The short name of an id ('' when it has none): a leaf's outcome name first when `leaf` (`LEAF_NAMES`: G6 the
    leaf 'too short to decide'); an angle rule by the word it set when `subtype` names one (`ANGLE_SUBTYPE_NAMES`: G6
    arc 'arc without turning points'); a path of the status tree part by part ('5d/6' -> 'force measured → judged by
    behavior'); a thin ledger entry 'U:<entry>' by its family U."""
    rule_id = str(rule_id)
    if leaf and rule_id in LEAF_NAMES:
        return LEAF_NAMES[rule_id]
    if not leaf and str(subtype or '') in ANGLE_SUBTYPE_NAMES.get(rule_id, {}):
        return ANGLE_SUBTYPE_NAMES[rule_id][str(subtype)]
    if rule_id in RULE_NAMES:
        return RULE_NAMES[rule_id]
    if rule_id.startswith('U:'):
        return RULE_NAMES['U']
    if '/' in rule_id:
        parts = [rule_name(part, leaf) for part in rule_id.split('/')]
        return ' → '.join(parts) if all(parts) else ''
    return ''


def named(rule_id: str) -> str:
    """'short name [id]': 'a carries the swing [B1]', 'earlier a-wave [cap:onset_failed]'; 'unmapped check [X]' for an
    id without a name."""
    rule_id = str(rule_id)
    shown = 'U' if rule_id.startswith('U:') else rule_id
    return f'{rule_name(rule_id) or UNMAPPED} [{shown}]'


def decided(rec_rule: str) -> str:
    """The subtitle's rule: 'decided: circulation (C)', 'decided: force measured → judged by behavior (5d/6)'; ''
    without a rule."""
    rec_rule = str(rec_rule or '')
    if not rec_rule:
        return ''
    return f'decided: {rule_name(rec_rule, leaf=True) or "unmapped rule"} ({rec_rule})'


# --------------------------------------------------------------------------------------------- caps


@dataclass(frozen=True)
class CapText:
    """How the column speaks of a cap: what it means (`words`) and the margin whose number explains it (`margin`)."""

    words: str
    margin: str = ''


# One row per cap of `confidence.Cap`; CAP_WORDS and CAP_MARGINS are views of it
CAPS = {
    'three_tp_episode': CapText('the libration rests on three turning points only', margin='N'),
    'exit_suspected': CapText('the angle seems to leave the resonance at the end'),
    'entry_suspected': CapText('the angle seems to enter the resonance at the start'),
    'transient_unexplained': CapText('no mechanism explains the transitions'),
    'one_return_cycle': CapText('the libration is one cycle closed by its phase return (two turning points)'),
    'force_law_disagrees': CapText('the force law fails while the potential holds'),
    'slow_libration_not_excluded': CapText('a slower libration is not excluded', margin='B5'),
    'planets_missing': CapText('the planets were not saved: no subresonance scan'),
    'owner_search_skipped': CapText('the owner search was switched off'),
    'catalogue_longitudes': CapText('the owner search used catalog planet longitudes (approximate)'),
    'nodal_subresonance_candidate': CapText('a nodal member may carry the libration'),
    'subresonance_tie': CapText('the member was chosen by its order only'),
    'subresonance_t6_only': CapText('the scan ran on the theory trigger alone'),
    'subresonance_theory_unconfirmed': CapText('the multiplet theory does not confirm the member'),
    'subresonance_undetermined': CapText('the members cannot be told apart (ϖ_P − ϖ librates)'),
    'jitter_low_e': CapText('the p·ϖ term wobbles by more than 45° at short periods (osculating)'),
    'all_physics_not_measured': CapText('none of force law, potential, zone, owner and strength could be measured'),
    'zone_not_measured': CapText('the separatrix zone was not measured'),
    'l2_without_zone': CapText('the amplitude exceeds the thin band without a measured zone', margin='L2'),
    'onset_failed': CapText(
        'a coherent a-wave at this period was already there before the episode and continues into it in phase; '
        "this resonance's own force does not explain it",
        margin='O1',
    ),
    'no_full_physics_support': CapText('no physical support (force law or 1:1 law, potential, strength) passes with its full margin'),
    'force_law_theory_only': CapText(
        "ȧ follows only the planet's averaged force of this term at the libration scale (theory, one fitted scale); "
        'the harmonic force law itself fails',
        margin='L4',
    ),
    'one_coorbital_leg': CapText(
        'the capture rests on one leg between two passages: a clean swing of one co-orbital well whose local 1:1 law, '
        'energy integral and two reflecting turns pass, but no closed cycle'
    ),
    'record_ends_unresolved': CapText(
        'after circulating, σ_f slows down at the end: since its last revolution it has stayed longer than that '
        'revolution took without completing another, and the slow center does not turn back'
    ),
    'slow_scale_provisional': CapText(
        'the slow libration was read at a shorter cutoff of the scale ladder; the force law (L4) and the potential '
        'with a (R0) are not measured there: only the theoretical strength supports it',
        margin='L4',
    ),
    'episodes_in_separatrix_zone': CapText(
        "every cycle lies in the model's separatrix zone and the angle crosses it: the edges of the librating "
        "stretches may be the zone's own crossings, not a capture or an escape"
    ),
    'near_separatrix_without_zone': CapText('the pattern alone places σ at the separatrix: the zone was not measured'),
    'geometry_only': CapText("neither the force law, the potential with a nor the strength measures this angle's own force"),
    'strength_scale_only': CapText(
        "neither the force law nor the potential with a measures this angle's own force; only the order-of-magnitude strength scale (μ_i μ_j e^q) agrees",
        margin='F',
    ),
    'implausible_strength': CapText("the verdict rests on the strength's order of magnitude, not on the angle's dynamics"),
    'episode_before_jump': CapText('the librations precede a jump of a; their potential and zone were not measured'),
    'episode_force_law_mixed': CapText('the force law holds on one librating episode and fails on another'),
    'near_episode': CapText('one more turning point would make a librating episode'),
    'short_period_aliased': CapText(
        'the output step folds short-period terms into slow ones (Δt ≥ P_orb/2 or P_syn/2, or a harmonic folded slower than the period floor)'
    ),
    'one_cycle_energy_not_conserved': CapText('the 1:1 libration is one cycle and its energy integral fails'),
    'action_step_provisional': CapText(
        'the libration was cut at a step of its action (G2a) with one energy cycle on a side, or with one cycle of the larger '
        'swings, whose period law is then read in sample'
    ),
    'second_capture_unresolved': CapText(
        'both resonances at once? is not ruled out: x = Ψ/u neither swept 180° without a potential nor followed the ' 'secular cycle alone'
    ),
}
CAP_WORDS = {cap: text.words for cap, text in CAPS.items() if text.words}
CAP_MARGINS = {cap: text.margin for cap, text in CAPS.items() if text.margin}
# the cap short_period_aliased named by the condition that fired (`_cap_detail`; the analysis' `short_period_trigger`)
SHORT_PERIOD_TITLES = {
    'nyquist': 'the output step folds short periods into slow ones (Δt {comparison} = min(P_orb, P_syn)/2)',
    'harmonic': 'the output step folds a harmonic of the short periods slower than the period floor',
}

# --------------------------------------------------------------------------------------------- margins and checks

MMR_MARGIN_NAMES = {
    'L1': 'enough turning points at P_c ×½, 1, 2',
    'L2': 'cycles away from the separatrix',
    'L3': 'dσ/dt = c (a − a*): the rate follows a',
    'L4': 'ȧ = K sin(σ − φ0): force law',
    'L5': 'one pendulum: T0 stable',
    'L6': 'center stationary',
    'L7': 'covers the record with margin',
    'L8a': 'the output step resolves σ',
    'L8b': 'the filter passes the cycle',
    'O1': 'no a-wave before the libration',
    'R0': 'energy fits the potential from a',
    'Z': 'separatrix zone measured',
    'S': 'no multiplet member holds the center',
    'W': 'no other resonance drives a',
    'F': 'strength plausible (×10 of theory)',
    'C1': 'center passes ≥ 10 rev',
    'C2': 'no turning points at any cutoff',
    'C3': 'dσ/dt smooth (one sign, no stalls)',
    'C4': 'a-wave as a circulation predicts',
    'C5': 'far from a* and one alias branch',
    'K1': 'slow σ follows p ϖ',
    'K1e': 'episode σ follows p ϖ, flat a, secular period',
    'B5': 'a slower libration excluded',
    'N': 'not one turning point short of an episode',
}
# the name of a sub-check whose measured comparison fails (its row ⚠, ✗ or –): MMR_MARGIN_NAMES states the passing
# condition, which beside failing numbers reads as their opposite ('centre stationary (5.77 > 0.5)'); the builder
# (`verdict_column_mmr._reading_name`) picks this name from the comparison the row prints, never from the icon alone
MMR_MARGIN_FAILED = {
    'L1': 'too few turning points at P_c ×½, 1, 2',
    'L2': 'cycles near the separatrix',
    'L3': 'dσ/dt ≠ c (a − a*): the rate does not follow a',
    'L4': 'ȧ does not follow K sin(σ − φ0)',
    'L4:coorbital': 'ȧ does not follow the 1:1 force law',
    'L5': 'T0 varies more than one pendulum allows',
    'L6': 'center moves',
    'L7': 'covers the record, short of the full margin',
    'L8a': 'the output step is too coarse for σ',
    'L8b': 'the filter weakens the cycle',
    'O1': 'a swings already before the libration',
    'R0': 'energy does not fit the potential from a',
    'Z': 'separatrix zone only estimated',
    'W': 'no driver accepted, one fits a in part',
    'F': 'strength outside ×10 of theory',
    'C1': 'center passes fewer whole turns than a clear circulation',
    'C4': 'a-wave unlike a circulation’s',
    'B5': 'a slower libration not excluded',
}
LONG_PERIOD_MARGIN_NAMES = {
    'N': 'more than three turning points',
    'S': 'center stationary (bands overlap)',
    'F': 'libration slower than 2 P_c (filter guard)',
    'P': 'center passes ≥ 1 turn at every smoothing',
    'G': 'free-ω gate: cycles to spare',
}
# Why K1 (the slow σ against p·ϖ) has no number: the codes of `mmr.kinematics.check_reason` (kin_check_reason, K1E_*)
# in words; {min_points} is MMR_KIN_MIN_POINTS (W28 E4, card g10-4b: 285637 printed 'not measured' without a reason)
KIN_REASON_WORDS = {
    'retrograde': 'no ϖ convention for i > 90°',
    'no_body_varpi_term': 'σ has no ϖ term of the body (p = 0)',
    'no_episode': 'the angle has no librating episode to read',
    'too_few_points': 'the record holds fewer than {min_points} independent center windows',
    'filter_not_run': 'the low-pass at the response cutoff could not run',
    'no_swing': 'neither σ nor p·ϖ moves',
}
CHECK_WORDS = {
    'slow_libration': 'B5 slower libration',
    'kinematic': 'ϖ-kinematic test',
    'exit': 'exit/entry',
    'one_pendulum': 'one pendulum',
    'onset': 'onset',
    'separatrix_zone': 'zone',
    'strength': 'strength',
    'identity': 'identity',
    'a_response': 'force law',
    'potential': 'potential',
    'filter': 'filter guard',
    'baseline': 'baseline',
    'alias': 'alias',
    'a_amplitude': 'B1',
}

# --------------------------------------------------------------------------------------------- reasons and flags

REJECT_WORDS = {
    'range': 'its center swings as far as σ’s',
    'rate': 'its center drifts',
    'pass': 'its center makes a full turn',
    'A_not_measured': 'its force phase was not measured',
    'A_coherence': 'its force phase is not fixed',
    'A_gain': 'it explains ȧ no better than σ',
    'B_centre': 'its center is not at 0 or π',
    'C_tie': 'another member fits as well',
    'D_theory': 'the multiplet theory disagrees',
    'E_t0': 'its T0 disagrees',
    'E_strength': "its strength factor is below the strongest term's share",
    'planets_missing': 'planets not saved',
    'undetermined': 'members indistinguishable',
    'nodal': 'a nodal member',
}
THEORY_WORDS = {
    'subresonance_theory_crossing': 'the orbits can cross',
    'subresonance_theory_round_off': 'coefficients below round-off',
    'subresonance_theory_range': 'e or e_P outside (0, 1)',
    'subresonance_theory_disagrees': 'measured and disagrees',
}
# own_flags -> why the owner search has no verdict (by design, or data missing; the state is `verdict_text`'s)
OWNER_FLAG_WORDS = {
    'owner_no_longitudes': "the planets' longitudes were not saved",
    'owner_too_few_windows': 'the libration holds too few force windows for the owner fit',
    'owner_no_period': 'no cycle period to cut the windows with',
    'owner_unwrap_failed': "a candidate's angle could not be unwrapped",
    'owner_search_off': 'the owner search is switched off',
    'owner_retrograde': 'no owner search for a retrograde angle',
    'owner_catalogue_ambiguous': 'a candidate on catalog λ (approx.) is too close to call',
}
OWNER_REASON_WORDS = {
    'phase_incoherent': 'its phase φ0 does not hold, coherence {coherence}',
    'unwrap_failed': 'its angle could not be unwrapped',
    'too_few_windows': 'too few force windows',
    'psi_share_high': 'σ − its angle keeps too much of the swing',
    'ripple_only': 'it explains only a ripple on σ',
    'not_better_than_sigma': 'it fits a no better than σ itself',
    'catalogue_ambiguous': 'on catalog longitudes its coherence sits at a threshold',
    'rate_outside_band': 'its forced wave would lie outside the a band',
    'phase_unresolved': 'the phases cannot tell it from σ',
    'potential_scale_out': 'its averaged potential needs a scale outside the band',
    'potential_unconfirmed': 'the averaged force disagrees with the fit',
}
# base_cause -> what is missing when the record already holds the time P needs
BASE_CAUSE_WORDS = {
    'too_few_tp': 'the run of turning points stops before a cycle closes',
    'centre_moves': 'the center moves along the run of turning points',
    'edge': 'the center moves along the run of turning points',
    'over_the_top': 'σ goes over the top next to the run of turning points',
    'no_closed_cycle': 'no cycle of the run of turning points returns about its center',
    'period_fits_record': 'the angle does not come back within it',
}
STRENGTH_REASON_WORDS = {
    'no_required_strength': 'no cycle to read a required strength from',
    'no_planet_orbit': "the planet's orbit is not known",
    'close_approach_at_centre': 'the body comes too close to the planet at the center: outside the averaged problem',
    'no_elements_in_window': 'no elements in the window of the snapshots',
    'no_stable_centre': 'the averaged potential has no stable center',
    'quadrature_not_converged': 'the averaged potential did not converge',
    'non_finite': 'a quadrature node fell on the planet',
    'precision_exhausted': 'the resonant harmonic is below the extended precision',
}
POTENTIAL_REASON_WORDS = {
    'off': 'the potential route is switched off',
    'three_body_owner': 'no direct theory for a three-body owner (second order)',
    'coorbital_owner': 'a 1:1 owner has its own force law',
    'retrograde': 'no averaged potential for a retrograde owner',
    'catalogue': "the planet's elements were not saved",
    'no_planet_mass': "the planet's mass is not known",
    'potential_cost': 'skipped for its cost (MMR_OWNER_POTENTIAL_MAX_KERNELS)',
    'not_converged': 'the averaged potential did not converge',
    'close_approach': 'the body comes too close to the planet',
    'no_windows': 'no window to evaluate it in',
}
EI_WORDS = {'zlk': 'ZLK (antiphase, constant H)', 'forced_beat': 'forced beat (in phase)', 'none': 'neither law'}

# --------------------------------------------------------------------------------------------- ledger entries and regimes

# Ledger entries of W20 (parts A and B) in words, by the start of their name (the longest match wins): the names stay
# the code's (columns, tests and the 'all' dump read them), the explained column prints these
LEDGER_WORDS = {
    '5 L4 null test': 'the force law beats a shifted in time (null test)',
    '5 L4 N-harmonic law: R2 gain': 'a second harmonic of σ improves the force law',
    '5 L4 N-harmonic law R2': 'the force law with harmonics of σ explains ȧ',
    '5 L4 N-harmonic law coefficients': 'the harmonics of the force law keep their phase from window to window',
    '5 L4 N-harmonic law null test': 'the harmonic force law beats ȧ shifted in time (full windows)',
    '5 L4 N-harmonic law on independent halves': 'the force law of one half of the windows predicts the other half',
    'cap short_period_aliased': 'the fast spread of σ_f against the swing (the step folds short periods)',
    'short_period_aliased: a harmonic folded': 'a harmonic of the short periods folded slower than the period floor',
    'short-period scatter of a': 'the short-period scatter of a against the libration’s a swing',
    '5 R-ALIAS-A folded step': 'the short-period scatter of a that reads a failed L4/L3 as not measured (R-ALIAS-A)',
    'G2 exit observed: the raw angle runs on': 'exit seen: the orbit-averaged σ runs on beyond the episode',
    'G2 exit observed: the raw angle moves monotonically': 'exit seen: the orbit-averaged σ moves one way',
    'G2 entry observed: the raw angle runs on': 'entry seen: the orbit-averaged σ ran in before the episode',
    'G2 entry observed: the raw angle moves monotonically': 'entry seen: the orbit-averaged σ moved one way',
    'G2c not when the exit is observed': 'no slower libration sought: the exit is seen at the fine scale',
    'G2c not when the entry is observed': 'no slower libration sought: the entry is seen at the fine scale',
    'G2 1:1 regimes without a circulation': 'the 1:1 regimes join without a circulation between them',
    'G2 1:1 range of the turning points': 'the turning points between the 1:1 regimes stay within a revolution',
    'G2 1:1 range of the raw angle': 'the raw angle between the 1:1 regimes stays within a revolution',
    'G2 accepted runs that are forced waves': 'the accepted runs are forced waves, not librations',
    'G2 forced wave': 'a run on a forced line of another argument',
    'G5a slow centre': 'the slow center turns back once: an arc, not a drift',
    'G1q the orbit-averaged angle stays': 'θ stays in the quasi-satellite well long enough',
    'G1q quasi-satellite floor': 'the quasi-satellite floor lies below the tadpole floor',
    'G1q the pass keeps': 'the quasi-satellite pass keeps the episodes outside the well',
    'G1q a stall with P_QS below the cutoff': 'a stall the cutoff hides: the quasi-satellite libration is faster',
    '2 owner route': 'the owner is accepted by its route',
    '2 owner cover of the decisive episode': 'the owner’s intervals cover σ’s decisive episode',
    '2 phase lever': 'the owner’s phase is steadier than σ’s own term could make it',
    '2 potential scale': 'the owner’s averaged force matches ȧ in scale',
    '2 second capture: x sweep': 'x = Ψ/u sweeps far enough to answer “both resonances at once?”',
    '2 second capture: energy R2': 'x = Ψ/u follows no potential (not both resonances at once)',
    '2 second capture: secular R2': 'ẍ of x = Ψ/u follows the secular harmonics (2ω, ϖ − ϖ_j)',
    '2 second capture: self R2': 'ẍ of x = Ψ/u follows no well of its own (sin x)',
    '3 K2 angle word': 'the angle word of the slow center',
    '3 K2b kinematic libration': 'the kinematic libration’s center moves more than its amplitude (a drift)',
    'G4b net shift': 'the slow center advances by whole revolutions net (circulation)',
    '3 kinematic (slow centre)': 'the slow center follows the perihelion term (K2)',
    '5s (before 5a/5e/5g/5u)': 'the strength, read before the force law',
    'G0 angle jitter': 'the p·ϖ term wobbles at short periods',
    'G0 full-angle residual': 'the full angle unwraps',
    'G0 varpi phase defined': 'ϖ itself is defined',
}
# what sigma does in a regime (final_regime, the last regime after a jump) for a sentence: no raw enum value there
REGIME_PHRASES = {
    'libration': 'σ librates',
    'episodic': 'σ librates in episodes',
    'circulation': 'σ circulates',
    'drift': 'σ drifts',
    'near_separatrix': 'σ moves near the separatrix',
    'insufficient_time': 'the record is too short to tell what σ does',
    'undefined': 'σ is not defined',
    'arc': 'σ_f slows to an arc (less than one revolution)',  # W28 C14: final_regime of an end after circulation
    'stall': 'σ_f stalls',
}

# the angle subtype (`AngleSubtype`) as the grey gloss of the σ key fact, only where it adds to the line's words
# (docs/physics/geometry.md); 'stationary' says the cycles' bands overlap, not that the centre stands still
ANGLE_SUBTYPE_WORDS = {
    'stationary': 'cycle bands overlap',
    'modulated': 'the center swings back and forth',
    'moving_centre': 'about a moving center',
    'regime_change': 'through a change of regime',
    'alias': 'folded by the output step',
    'arc': 'one arc, no closed cycle',
    'stall': 'σ stalls',
    'few_cycles': 'too few turning points for a cycle',
    'pattern_a': 'slips over the top by whole turns',
    'pattern_b': 'crawls over the top of its well, slower than a free rotation, in most revolutions',
    'undefined_phase': 'the phase is not defined',
}


# The panels of the chip 'glance at (…)' (decision D5, 2026-11-04; the author reviews the map once): for each input of
# the level that holds it (`policy.level_inputs` names), the panels that show it. MMR letters: docs/reading-figures.md
# (a angle, b σ mod 360°, d a, e response of a, h cycle energy, i centre, j related centres, l episode portrait);
# secular/ZLK: (a) angle, (b) mod 360°, (e)/(f) e and i, (g) the rate or the free ω, (h) centre, (j) free e vector.
GLANCE_PANELS = {
    'mmr': {
        'N': ('a', 'l'),
        'core': ('a', 'l'),
        'M': ('h',),
        'S': ('b', 'i'),
        'V': ('e', 'j'),
        'F': ('e', 'h'),
        'veto': ('j',),
        'second_capture': ('j',),
        'candidate': ('a',),
        'circulation': ('b', 'i'),
        'table': ('j',),
        'status': ('h',),
        'thin_veto': ('j',),
    },
    'long': {
        'N': ('a',),
        'core': ('a',),
        'M': ('h',),
        'S': ('b', 'h'),
        'V': ('g', 'j'),
        'veto': ('g', 'j'),
        'candidate': ('a',),
        'circulation': ('b', 'h'),
        'table': ('a',),
        'status': ('h',),
        'thin_veto': ('g',),
        'end_travel': ('a', 'b'),  # σ_f's travel to the record's end (W36 F5)
    },
}
# the gate detail of a ZLK row (zlk_gate_detail) -> the panels of its inputs instead: an unfitted co-orbital's free ω
# (g) and free e vector (j) are the split the column calls no measurement, so its V points at the episodes (a) and at
# a with the co-orbital zone (d)
GLANCE_OVERRIDES = {
    'coorbital_unfitted_forcing': {'V': ('a', 'd')},
}

# --------------------------------------------------------------------------------------------- caveats (right column)

# The Caveats block of the right column (`verdict_column`, decision of 2026-11-04 on the flags table of the spec review):
# flag -> (the statuses it is said on, the sentence). A flag outside this table, or on another status, goes to the
# footer only (the diagnostics: zone_eq_only, jump_locator_aliased, ...); the level inputs (measured_objection,
# veto_thin, second_capture_unresolved, filter_sensitive, owner_probable, ...) are pills of the Confidence block, never
# caveats. Slots: {jump} the start of the regime after the jump of a (regime_start_yr), {ratio} the short-period scatter
# of a over the episode's a swing (smp_da_sp_over_a_excursion), {exit_P} / {entry_P} the record left after the exit /
# before the entry in the episode's periods (exit_remaining_P; entry_remaining_P, else record_entry_remaining_P after a
# cut); a sentence whose slot has no value is not said.
LIBRATING = ('resonant', 'transient', 'near_separatrix')
ALL_STATUSES = ('resonant', 'transient', 'near_separatrix', 'non_resonant', 'uncertain', 'invalid')
FLAG_CAVEATS = {
    'episodes_in_separatrix_zone': (
        ('transient',),
        'Every libration cycle lies in the separatrix zone and σ crosses it, so the episode edges may be crossings of '
        'the chaotic layer rather than captures and escapes <<h>>.',
    ),
    'transient_unexplained': (
        ('transient',),
        'No mechanism of the separatrix model explains the switches between libration and circulation.',
    ),
    'libration_in_separatrix_zone': (
        ('resonant',),
        'Libration cycles lie in the separatrix zone: the libration runs in the chaotic layer near the separatrix <<h>>.',
    ),
    'near_separatrix_without_zone': (
        ('near_separatrix',),
        'Only the pattern of σ puts it at the separatrix; the separatrix zone was not measured.',
    ),
    'exit_suspected': (
        LIBRATING,
        'The exit lies {exit_P} libration periods before the record’s end, so the libration may just continue past the record.',
    ),
    'entry_suspected': (
        LIBRATING,
        'The entry lies {entry_P} libration periods after the record’s start, so the libration may have begun before the record.',
    ),
    'last_regime_after_jump': (
        ALL_STATUSES,
        'a jumps at {jump}; the record before and after the jump is measured as two regimes, and the one after it gives '
        'the final regime.',
    ),
    'episode_before_jump': (
        ALL_STATUSES,
        'The deciding episodes come before the jump of a at {jump}; their energy was not measured, since the potential '
        'belongs to the later regime.',
    ),
    'a_short_period_scatter': (
        LIBRATING,
        'The output step folds short-period terms, and their scatter in a is {ratio}× the episode’s a swing: the checks '
        'that read a may read folded lines.',
    ),
    'strength_scale_only': (
        ALL_STATUSES,
        'The angle’s own force is shown only by the order-of-magnitude strength scale μ_i μ_j e^q, not by the force law '
        'or the potential.',
    ),
    # true whether no force check passed or each passed near its threshold; the level reads only whether one passed
    # (policy F), so the sentence says so; not said beside strength_scale_only, which names the one support
    'no_full_physics_support': (
        ALL_STATUSES,
        'No check of the angle’s own force passed with a full margin; the level reads only whether one passed at all.',
    ),
    'force_law_theory_only': (
        ALL_STATUSES,
        'ȧ follows only the computed averaged force of this term; the measured force law itself fails.',
    ),
    'slow_scale_provisional': (
        ALL_STATUSES,
        'The slow libration was read at a shorter smoothing cutoff where neither the force law nor the potential was measured.',
    ),
    'force_law_disagrees': (ALL_STATUSES, 'The energy fits the potential, but ȧ does not follow sin σ.'),
    'episode_force_law_mixed': (
        ALL_STATUSES,
        'The force law holds on some episodes and fails on others: not every episode may belong to this resonance.',
    ),
    'one_cycle_energy_not_conserved': (
        ALL_STATUSES,
        'The single 1:1 cycle breaks the co-orbital energy integral: it may be a passage, not motion in the averaged well.',
    ),
    'subresonance_composite': (
        ALL_STATUSES,
        'Which multiplet member librates is not settled: a composite of members dominates.',
    ),
    'subresonance_tie': (ALL_STATUSES, 'Which multiplet member librates is not settled: two members tie, chosen by order.'),
    'subresonance_undetermined': (ALL_STATUSES, 'Which multiplet member librates is not settled: the members are indistinguishable.'),
    'nodal_subresonance_candidate': (ALL_STATUSES, 'Which multiplet member librates is not settled: a nodal member is possible.'),
    'subresonance_theory_disagrees': (
        ALL_STATUSES,
        'Which multiplet member librates is not settled: the multiplet theory disagrees with the measured member.',
    ),
    'action_step_provisional': (
        ALL_STATUSES,
        'The step of the libration’s action rests on one energy cycle on a side: the cut between the regimes is provisional.',
    ),
    'catalogue_longitudes': (
        ALL_STATUSES,
        'The other resonances were searched on catalog planet longitudes, not on saved ones (approximate).',
    ),
    'owner_search_skipped': (ALL_STATUSES, 'Another resonance driving a was not searched (the owner search is off).'),
    'planets_missing': (ALL_STATUSES, 'The planets were not saved: the tests that need their longitudes did not run.'),
}
# the flags whose caveat is said only when the status rests on a multiplet member (rule 1)
MEMBER_CAVEATS = (
    'subresonance_composite',
    'subresonance_tie',
    'subresonance_undetermined',
    'nodal_subresonance_candidate',
    'subresonance_theory_disagrees',
)
