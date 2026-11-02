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
- the words of reasons, flags, ledger entries, margins and regimes (`*_WORDS`, `*_NAMES`, `REGIME_PHRASES`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

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
    'G2c': 'slow-centre libration',
    'G3': 'stalls and slips at the separatrix',
    'G4': 'centre makes a revolution',
    'G4b': 'net revolutions of the centre',
    'G4c': 'single co-orbital leg',  # W28 C6: a 1:1 circulation judged on its best single leg
    'G4h': 'one half-swing between passages',  # W28 C6, q-5 i: not 1:1
    'G5': 'drifting centre',
    'G5a': 'one slow arc',
    'G5w': 'slow swing in the well',
    'G6': 'too few cycles',
    # the leaves of the status tree (mmr/decide.py decide_status, the column rec_rule)
    'C': 'circulation',
    '1': 'a multiplet member librates',
    '2': 'another resonance drives a',
    '3': 'kinematic libration',
    '3k2': 'kinematic drift of the centre',
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
    '6': 'judged by behaviour',
    '6u': 'union covers the record',
    '2j': 'both resonances? one cycle of x',  # W28 A2b: 'both resonances at once?' with one cycle of x -> extend
    'K3s': 'slow scale is a secular cycle',  # W28 B10a: a circulation whose slow scale is a secular cycle
    # margins and checks (mmr/decide.py MARGIN_CLASSES, the tree's checks)
    'L1': 'survives the filter',
    'L2': 'away from the separatrix',
    'L3': 'σ̇ follows a − a*',
    'L4': 'a obeys the force law',
    'L5': 'one pendulum',
    'L6': 'fixed centre',
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
    'S': 'no member holds the centre',
    'W': 'no other resonance drives a',
    'F': 'strength plausible',
    'N': 'not short of an episode',
    'F6': 'forced wave of another angle',
    'F6r': 'unconfirmed ripple',
    'F7': 'owner covers the episode',
    'C1': 'centre passes whole revolutions',
    'C2': 'no reversals at any cutoff',
    'C3': 'steady rate',
    'C4': 'too fast for a pendulum',
    'C5': 'far from resonance',
    'K1': 'slow σ follows pϖ',
    'K1e': 'episode σ follows pϖ, a flat',
    'K2': 'slow centre follows ϖ',
    'B5': 'slower libration excluded',
    'U': 'near its convention',  # a ledger comparison in the thin band of a convention (U:<entry>)
    # secular and Lidov-Kozai margins (secular/decide.py); SZ. tells them from the MMR margins N, S, F
    'SZ.N': 'enough reversals',
    'SZ.S': 'centres agree',
    'SZ.F': 'slower than the cutoff',
    'SZ.P': 'circulates at every smoothing',
    'SZ.G': 'free-ω gate',
}
# A leaf that is the outcome of a check of the same id: the subtitle names the outcome ('decided: no resonant
# potential (R0)'), the check row the check ('energy fits the potential [R0]'). C5 decides non_resonant or invalid.
LEAF_NAMES: Dict[str, str] = {
    'R0': 'no resonant potential',
    'C5': 'aliased circulation',
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
NAME_MAX = 40  # characters of a short name: the column is about 60 wide, and the id follows
UNMAPPED = 'unmapped check'  # an id without a name is printed as such, never guessed (Codex, card g10-3)
LONG_PERIOD_PREFIX = 'SZ.'


def long_period_id(key: str) -> str:
    """The printed id of a secular/ZLK margin: 'N' -> 'SZ.N'."""
    return f'{LONG_PERIOD_PREFIX}{key}'


def rule_name(rule_id: str, leaf: bool = False) -> str:
    """The short name of an id ('' when it has none): a leaf's outcome name first when `leaf` (`LEAF_NAMES`); a path
    of the status tree part by part ('5d/6' -> 'force measured → judged by behaviour'); a thin ledger entry
    'U:<entry>' by its family U."""
    rule_id = str(rule_id)
    if leaf and rule_id in LEAF_NAMES:
        return LEAF_NAMES[rule_id]
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
    """The subtitle's rule: 'decided: circulation (C)', 'decided: force measured → judged by behaviour (5d/6)'; ''
    without a rule."""
    rec_rule = str(rec_rule or '')
    if not rec_rule:
        return ''
    return f'decided: {rule_name(rec_rule, leaf=True) or "unmapped rule"} ({rec_rule})'


# --------------------------------------------------------------------------------------------- caps


@dataclass(frozen=True)
class CapText:
    """How the explained column speaks of a cap: what it means (`words`), the other verdict the record may hold and
    what on the figure decides between the two (`alternative`, `decides`; author 2026-10-19 q1: medium is a spot check,
    so a cap that lowers the level names the alternative and where to look; the panel letters are those of the
    combined figure, `mmr_combined.PANELS`), and the margin whose number explains it (`margin`, printed with the cap
    in the confidence sentence)."""

    words: str
    alternative: str = ''
    decides: str = ''
    margin: str = ''


# One row per cap (every cap of `confidence.Cap` has an alternative: tested); CAP_WORDS, CAP_ALTERNATIVES and
# CAP_MARGINS are views of it
CAPS = {
    'three_tp_episode': CapText(
        'the libration rests on three turning points only',
        'no libration yet (insufficient_time): three turning points are one cycle',
        '(a): whether σ_f closes a second cycle about the same centre',
        margin='N',
    ),
    'exit_suspected': CapText(
        'the angle seems to leave the resonance at the end',
        'the libration goes on past the end of the record (or σ leaves for good)',
        '(a) and (i) at the right edge: whether σ_f turns back once more or runs a full revolution after the last turn',
    ),
    'entry_suspected': CapText(
        'the angle seems to enter the resonance at the start',
        'the libration started before the record (or σ is captured only here)',
        '(a) and (i) at the left edge: whether σ_f runs a full revolution before the first turn',
    ),
    'transient_unexplained': CapText(
        'no mechanism explains the transitions',
        'near_separatrix: σ wanders across the separatrix without a capture or an escape',
        '(h): h of the cycles next to each transition against the zone and h = 1; (c): the slips',
    ),
    'one_return_cycle': CapText(
        'the libration is one cycle closed by its phase return (two turning points)',
        'no libration yet (insufficient_time): a half-swing that happened to come back',
        '(a): the one cycle between its two turning points; σ_f must cross the return level back at the rate it left',
    ),
    'force_law_disagrees': CapText(
        'the force law fails while the potential holds',
        'uncertain (origin unknown): the potential holds while ȧ does not follow sin σ',
        '(e): ȧ against R sin(σ_f − φ0) per window and the R² of the halves',
    ),
    'slow_libration_not_excluded': CapText(
        'a slower libration is not excluded',
        'a slower libration of σ about a moving centre (the verdict of the slow scale)',
        '(i): whether the slow centre of σ_f turns back inside the record',
        margin='B5',
    ),
    'planets_missing': CapText(
        'the planets were not saved: no subresonance scan',
        'non_resonant: a member of the multiplet holds the centre (subresonance)',
        '(j) has no members to draw: rerun with the planets saved',
    ),
    'owner_search_skipped': CapText(
        'the owner search was switched off',
        'non_resonant: another resonance drives a (induced)',
        '(b) and (j) draw no owner: rerun with the owner search on',
    ),
    'catalogue_longitudes': CapText(
        'the owner search used catalogue planet longitudes (approximate)',
        'another owner, or none, with the true longitudes of the planets',
        "(b): the owner's angle against σ; rerun with the planets saved",
    ),
    'nodal_subresonance_candidate': CapText(
        'a nodal member may carry the libration',
        'non_resonant: a nodal member of the multiplet holds the centre',
        "(j): the nodal member's centre against σ's",
    ),
    'subresonance_tie': CapText(
        'the member was chosen by its order only',
        'another member of the same order holds the centre',
        '(j): the centres of the tied members, the flatter one owns the libration',
    ),
    'subresonance_t6_only': CapText(
        'the scan ran on the theory trigger alone',
        'no member holds the centre: σ itself librates',
        "(j): whether the member's centre is flatter than σ's",
    ),
    'subresonance_theory_unconfirmed': CapText(
        'the multiplet theory does not confirm the member',
        'σ itself, or another member, holds the centre',
        "(j): the member's centre against the multiplet theory's line",
    ),
    'subresonance_undetermined': CapText(
        'the members cannot be told apart (ϖ_P − ϖ librates)',
        'another member of the multiplet holds the centre (they move together while ϖ_P − ϖ librates)',
        '(j): the members whose centres stay flat with σ',
    ),
    'jitter_low_e': CapText(
        'the p·ϖ term wobbles by more than 45° at short periods (osculating)',
        'invalid (undefined phase): ϖ itself is undefined at small e',
        '(b): the spread of the osculating σ about σ_f',
    ),
    'all_physics_not_measured': CapText(
        'none of force law, potential, zone, owner and strength could be measured',
        'uncertain: nothing measured shows this angle’s own force',
        '(e) and (h) have no force law or energy to read: only the shape of σ in (a) decides',
    ),
    'zone_not_measured': CapText(
        'the separatrix zone was not measured',
        'near_separatrix: the cycles may sit in the separatrix zone',
        '(h): h of each cycle against h = 1',
    ),
    'l2_without_zone': CapText(
        'the amplitude exceeds the thin band without a measured zone',
        'near_separatrix: the large swings may cross the separatrix',
        '(h): h of the widest cycles against h = 1',
        margin='L2',
    ),
    'onset_failed': CapText(  # B4 (cards g03-4 A, g03-5 A): only a coherent, in-phase wave the own law does not explain
        'a coherent a-wave at this period was already there before the episode and continues into it in phase; '
        "this resonance's own force does not explain it",
        'non_resonant: σ rides an a-wave of another origin that began before the episode',
        '(d): the a-wave before the episode against the one inside it',
        margin='O1',
    ),
    'no_full_physics_support': CapText(  # q-2 b, author 2026-10-31
        'no physical support (force law or 1:1 law, potential, strength) passes with its full margin',
        "the oscillation is not this angle's own: thin evidence fits another origin as well",
        '(e) and (h): the force law and the energy fit, each only thin',
    ),
    'force_law_theory_only': CapText(  # W28 B8, card g07-2 A
        "ȧ follows only the planet's averaged force of this term at the libration scale (theory, one fitted scale); "
        'the harmonic force law itself fails',
        "uncertain (origin unknown): the slow ȧ fits the theory, but a rival force hidden by the smoothing is not excluded",
        '(e): the averaged force of the term against ȧ, both smoothed at the libration scale',
        margin='L4',
    ),
    'one_coorbital_leg': CapText(  # W28 C6, leaf G4c, q-4 A
        'the capture rests on one leg between two passages: a clean swing of one co-orbital well whose local 1:1 law, '
        'energy integral and two reflecting turns pass, but no closed cycle',
        'uncertain: half a cycle that the planet happened to reverse twice, no capture',
        '(a): the leg between its two turns; (e) the local 1:1 law on the leg; (h) |a - a_P| before and after each turn',
    ),
    'record_ends_unresolved': CapText(  # W28 C14, card g01-7 A, rule end_arc_extend
        'after circulating, σ_f slows down at the end: since its last revolution it has stayed longer than that '
        'revolution took without completing another, and the slow centre does not turn back',
        'a capture at the end of the record (an episode beyond it): the slowdown is the start of a libration',
        '(a) at the right edge: σ_f after its last full revolution; a longer integration forward (flag extend_time) decides',
    ),
    'slow_scale_provisional': CapText(  # W28 B7, card g07-4 A
        'the slow libration was read at a shorter cutoff of the scale ladder; the force law (L4) and the potential '
        'with a (R0) are not measured there: only the theoretical strength supports it',
        'L4 untested; a record ≥ 5.25 P decides (with fewer cycles the libration may still be another origin)',
        '(a): the slow σ_f at the ladder cutoff; (e) has no force-law windows yet',
        margin='L4',
    ),
    'episodes_in_separatrix_zone': CapText(
        (
            "every cycle lies in the model's separatrix zone and the angle crosses it: the edges of the librating "
            "stretches may be the zone's own crossings, not a capture or an escape"
        ),
        'near_separatrix: the edges of the librating stretches are crossings of the zone, not a capture or an escape',
        '(h): h of the cycles at each edge inside the blue band',
    ),
    'near_separatrix_without_zone': CapText(
        'the pattern alone places σ at the separatrix: the zone was not measured',
        'non_resonant circulation, or a libration close to the separatrix',
        '(c): the slips; (h): h near 1',
    ),
    'geometry_only': CapText(
        "neither the force law, the potential with a nor the strength measures this angle's own force",
        'uncertain: only the shape of σ says libration',
        '(e) and (h): no force law and no energy were measured',
    ),
    'strength_scale_only': CapText(
        "neither the force law nor the potential with a measures this angle's own force; only the order-of-magnitude strength scale (μ_i μ_j e^q) agrees",
        'uncertain: the scale agrees by an order of magnitude only, the angle’s own force is not computed',
        '(e) and (h): no force law and no energy were measured; the physics block: μ_req against μ_i μ_j e^q',
        margin='F',
    ),
    'implausible_strength': CapText(
        "the verdict rests on the strength's order of magnitude, not on the angle's dynamics",
        'a libration whose force the first-order average misses',
        'the physics block: μ_req against μ_th, and whether a forced wave explains the cycles',
    ),
    'episode_before_jump': CapText(
        'the librations precede a jump of a; their potential and zone were not measured',
        'the libration ended with the jump of a: the episodes belong to the earlier regime',
        '(a) and (d): the episodes before the hatched jump (their energy is not measured)',
    ),
    'episode_force_law_mixed': CapText(
        'the force law holds on one librating episode and fails on another',
        'only the witness episode librates in this resonance, the others carry another force',
        '(e): R² of the halves per episode',
    ),
    'near_episode': CapText(
        'one more turning point would make a librating episode',
        'transient: one more turning point would make a librating episode',
        '(a): the grey run with most turning points',
    ),
    'short_period_aliased': CapText(
        'the output step folds short-period terms into slow ones (Δt ≥ P_orb/2 or P_syn/2, or a harmonic folded slower than the period floor)',
        'the turning points are partly the step’s: short-period terms folded into σ_f and a_f',
        '(a): the fast spread about σ_f; a rerun at a step that folds no short period decides',
    ),
    'one_cycle_energy_not_conserved': CapText(
        'the 1:1 libration is one cycle and its energy integral fails',
        'no co-orbital libration yet: the one cycle may be a passage, not motion in the averaged well',
        '(h): the energy of the cycle against the averaged potential; a longer record with a second cycle decides',
    ),
    'action_step_provisional': CapText(
        'the libration was cut at a step of its action (G2a) with one energy cycle on a side, or with one cycle of the larger '
        'swings, whose period law is then read in sample',
        'one libration over the whole window: the step is a large modulation, not two energy levels',
        '(a): the half-amplitudes on each side of the dotted cut; (h): h of the cycles before and after it',
    ),
    'second_capture_unresolved': CapText(
        'both resonances at once? is not ruled out: x = Ψ/u neither swept 180° without a potential nor followed the ' 'secular cycle alone',
        'two resonances at once: σ is held by its own term as well (resonant, not only induced)',
        '(j): Ψ/u = σ − (v/u) φ — a sweep of 180° or more that follows no potential, or ẍ that follows the secular '
        'cycle and not sin x, excludes it',
    ),
}
CAP_WORDS = {cap: text.words for cap, text in CAPS.items() if text.words}
CAP_ALTERNATIVES = {cap: (text.alternative, text.decides) for cap, text in CAPS.items() if text.alternative}
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
    'L6': 'centre stationary',
    'L7': 'covers the record',
    'L8a': 'the output step resolves σ',
    'L8b': 'the filter passes the cycle',
    'O1': 'no a-wave before the libration',
    'R0': 'energy fits the potential from a',
    'Z': 'separatrix zone measured',
    'S': 'no multiplet member holds the centre',
    'W': 'no other resonance drives a',
    'F': 'strength plausible (×10 of theory)',
    'C1': 'centre passes ≥ 10 rev',
    'C2': 'no turning points at any cutoff',
    'C3': 'dσ/dt smooth (one sign, no stalls)',
    'C4': 'a-wave as a circulation predicts',
    'C5': 'far from a* and one alias branch',
    'K1': 'slow σ follows p ϖ',
    'K1e': 'episode σ follows p ϖ, flat a, secular period',
    'B5': 'a slower libration excluded',
    'N': 'not one turning point short of an episode',
}
LONG_PERIOD_MARGIN_NAMES = {
    'N': 'more than three turning points',
    'S': 'centre stationary (bands overlap)',
    'F': 'libration slower than 2 P_c (filter guard)',
    'P': 'centre passes ≥ 1 rev at every smoothing',
    'G': 'free-ω gate: cycles to spare',
}
# Why K1 (the slow σ against p·ϖ) has no number: the codes of `mmr.kinematics.check_reason` (kin_check_reason, K1E_*)
# in words; {min_points} is MMR_KIN_MIN_POINTS (W28 E4, card g10-4b: 285637 printed 'not measured' without a reason)
KIN_REASON_WORDS = {
    'retrograde': 'no ϖ convention for i > 90°',
    'no_body_varpi_term': 'σ has no ϖ term of the body (p = 0)',
    'no_episode': 'the angle has no librating episode to read',
    'too_few_points': 'the record holds fewer than {min_points} independent centre windows',
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
    'range': 'its centre swings as far as σ’s',
    'rate': 'its centre drifts',
    'pass': 'its centre makes a full turn',
    'A_not_measured': 'its force phase was not measured',
    'A_coherence': 'its force phase is not fixed',
    'A_gain': 'it explains ȧ no better than σ',
    'B_centre': 'its centre is not at 0 or π',
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
    'owner_catalogue_ambiguous': 'a candidate on catalogue λ (approx.) is too close to call',
}
OWNER_REASON_WORDS = {
    'phase_incoherent': 'its phase φ0 does not hold, coherence {coherence}',
    'unwrap_failed': 'its angle could not be unwrapped',
    'too_few_windows': 'too few force windows',
    'psi_share_high': 'σ − its angle keeps too much of the swing',
    'ripple_only': 'it explains only a ripple on σ',
    'not_better_than_sigma': 'it fits a no better than σ itself',
    'catalogue_ambiguous': 'on catalogue longitudes its coherence sits at a threshold',
    'rate_outside_band': 'its forced wave would lie outside the a band',
    'phase_unresolved': 'the phases cannot tell it from σ',
    'potential_scale_out': 'its averaged potential needs a scale outside the band',
    'potential_unconfirmed': 'the averaged force disagrees with the fit',
}
# G2a test 7's sub-tests (`geometry.T7_*`) in words: why one potential does not hold both regimes
T7_WORDS = {
    'isochronous': 'both regimes keep one period, as a forced wave would (the period law cannot tell)',
    'shape': 'V of the larger swings explains too little of their σ̇²',
    'covered': 'the smaller swings leave the range the larger ones sampled',
    'no_loop': 'a cycle has no loop of its energy in V',
    'period_law': 'a cycle\'s period is not the loop\'s',
    'turning_points': 'a cycle swings off the turning points of its loop',
    'energy_drift': 'a side\'s energy does not stay put',
    'wells_unstable': 'V\'s wells change with the shape cutoff',
}
# base_period_source -> where the period P of the baseline comes from (`decide.baseline_period`)
BASE_SOURCE_WORDS = {
    'tp_spacing': 'twice the median spacing of the turning points',
    'arc_slow': 'twice the longest stretch of the slow centre without a turn',
    'two_omega': 'the 2ω cycle the angle follows',
    'modulation': 'the e-cycle 2π/|g − g5| the angle follows',
    'stall_rate': 'one revolution at the rate of the stall',
    'stall_force': 'the shortest libration period the force bound of the stall allows',
    'cycle': 'the measured cycle',
    'b5': 'the shortest libration period B5 allows',
}
# base_cause -> what is missing when the record already holds the time P needs
BASE_CAUSE_WORDS = {
    'too_few_tp': 'the run of turning points stops before a cycle closes',
    'centre_moves': 'the centre moves along the run of turning points',
    'edge': 'the centre moves along the run of turning points',
    'over_the_top': 'σ goes over the top next to the run of turning points',
    'no_closed_cycle': 'no cycle of the run of turning points returns about its centre',
    'period_fits_record': 'the angle does not come back within it',
}
STRENGTH_REASON_WORDS = {
    'no_required_strength': 'no cycle to read a required strength from',
    'no_planet_orbit': "the planet's orbit is not known",
    'close_approach_at_centre': 'the body comes too close to the planet at the centre: outside the averaged problem',
    'no_elements_in_window': 'no elements in the window of the snapshots',
    'no_stable_centre': 'the averaged potential has no stable centre',
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
GATE_WORDS = {
    'confirmed': 'the free ω librates too: a confirmed Lidov-Kozai libration',
    'kinematic': 'the free ω winds, so the osculating libration is that of the forced eccentricity: kinematic',
    'no_verdict': 'the gate gave no verdict',
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
    'G5a slow centre': 'the slow centre turns back once: an arc, not a drift',
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
    '3 K2 angle word': 'the angle word of the slow centre',
    '3 K2b kinematic libration': 'the kinematic libration’s centre moves more than its amplitude (a drift)',
    'G4b net shift': 'the slow centre advances by whole revolutions net (circulation)',
    '3 kinematic (slow centre)': 'the slow centre follows the perihelion term (K2)',
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

# the marks of rule regime_cut_mark (W28 C16, q-5 i; `geometry.MARK_CENTRE_STEP`, `MARK_STALL`, episodes.csv `marks`):
# one episode of an angle that is not 1:1 joined across a step or a stall of its centre (`verdict_text.regime_change_clause`)
REGIME_MARK_WORDS = {
    'centre_step': 'a step of its centre',
    'stall': 'a stall of its centre',
}
