"""Rule flags: which named rules acted on a row (summary column `rules_fired`), even when its verdict stayed the same.

A rule fires where its predicate holds and it acts on an intermediate result: a cut, a merge, a cap removed or added,
an outcome. Its owner calls `fire` there; a rule that was only evaluated does not fire. Geometry-level rules go into
`Geometry.rules_fired`, the other result objects carry frozensets of their own, and the decide steps merge them all
into `Verdict.rules_fired` (`ordered`: deduplicated, in registry order). The review folder of W28 (card s-4,
`same_rule/`) is built from this column.

The flags are no settings (settings_hash does not read them) and change no status, level, column value or figure.
`Rule` names a rule by its meaning; `RuleInfo.doc_id` is the id of the row of docs/libration-analysis.md that
describes the rule (several rules may share a row; `PENDING` for a new rule whose row is not written yet), the id
printed in brackets next to the rule's name (card g10-3).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable


class Rule(StrEnum):
    # owner and two resonances
    OWNER_PER_EPISODE = 'owner_per_episode'
    OWNER_PSI_TURNS = 'owner_psi_turns'
    SECOND_CAPTURE_F3 = 'second_capture_f3'
    BOTH_RESONANCES_CONFIRMED = 'both_resonances_confirmed'
    # joins and episodes
    FRAME_JOIN = 'frame_join'
    STALL_JOIN = 'stall_join'
    REGIME_CUT_MARK = 'regime_cut_mark'  # not 1:1: a centre step or stall inside one libration region (q-5 i)
    BOUNDED_GAP_MERGE = 'bounded_gap_merge'
    LOCAL_CENTRE_ENTRY = 'local_centre_entry'
    # B series
    FORCED_STRENGTH_MARK = 'forced_strength_mark'
    O1_FOREIGN_WAVE = 'o1_foreign_wave'
    R0_WINDOWS = 'r0_windows'
    POTENTIAL_ON_ARC = 'potential_on_arc'
    SUBFLOOR_PASS = 'subfloor_pass'
    ORBIT_CUTOFF = 'orbit_cutoff'
    RIPPLE_ON_ARC = 'ripple_on_arc'
    SLOW_SWING_IN_WELL = 'slow_swing_in_well'
    SCALE_LADDER = 'scale_ladder'
    DIRECT_FORCE_LAW = 'direct_force_law'
    EXTEND_LENGTH_LIMITED = 'extend_length_limited'
    EQ_SCALE_NOT_OBJECTION = 'eq_scale_not_objection'
    SLOW_SCALE = 'slow_scale'
    CENTRE_SCALE_DISPLACED = 'centre_scale_displaced'  # G2c guard: a contradicted center-scale reading keeps the fine one
    SECULAR_CYCLE_KINEMATIC = 'secular_cycle_kinematic'
    KINEMATIC_HIGH = 'kinematic_high'
    # subresonances
    SUBRESONANCE_THEORY_SELECT = 'subresonance_theory_select'
    SUBRESONANCE_T7 = 'subresonance_t7'
    SUBRESONANCE_T8 = 'subresonance_t8'
    # co-orbitals
    COORBITAL_FLOOR_PROFILE = 'coorbital_floor_profile'
    COORBITAL_QS_RUN_FLOOR = 'coorbital_qs_run_floor'
    COORBITAL_PAIR_LOOP_FLOOR = 'coorbital_pair_loop_floor'
    COORBITAL_EDGE_TURN = 'coorbital_edge_turn'
    QS_TRIGGER = 'qs_trigger'
    COORBITAL_LEG = 'coorbital_leg'
    COORBITAL_NEAR_EPISODE = 'coorbital_near_episode'
    COORBITAL_WELL_CUT = 'coorbital_well_cut'
    COORBITAL_WELL_CORE = 'coorbital_well_core'
    COORBITAL_RESIDENCE = 'coorbital_residence'
    SINGLE_HALF_LIBRATION = 'single_half_libration'  # not 1:1: one half-swing between passages (q-5 i)
    ELIGIBLE_REASON = 'eligible_reason'
    END_ARC_EXTEND = 'end_arc_extend'
    # secular and ZLK
    FREE_RESAMPLE_REMAINDER = 'free_resample_remainder'
    ZLK_INVARIABLE_FRAME = 'zlk_invariable_frame'
    ZLK_HALF_TURN_CUT = 'zlk_half_turn_cut'
    ZLK_COORBITAL_EPISODIC = 'zlk_coorbital_episodic'
    CENTRE_TURNED_INSUFFICIENT = 'centre_turned_insufficient'
    PHASE_UNRESOLVED = 'phase_unresolved'
    ZLK_GATE_UNCONFIRMED = 'zlk_gate_unconfirmed'


PENDING = 'pending'  # the doc id of a new rule whose row in docs/libration-analysis.md is not written yet


@dataclass(frozen=True)
class RuleInfo:
    doc_id: str  # the id of the rule's row in docs/libration-analysis.md, or PENDING
    title: str  # what the rule does, in a few words (the work package that owns it in brackets)
    since: str  # the iteration that introduced it


def _w28(doc_id: str, title: str) -> RuleInfo:
    return RuleInfo(doc_id, title, 'W28')


# The registry, in its print order (`ordered`): every Rule has exactly one entry (tested).
RULES: dict[Rule, RuleInfo] = {
    Rule.OWNER_PER_EPISODE: _w28('F7', 'the owner is credited only with its measured regimes, per accepted episode (A1)'),
    Rule.OWNER_PSI_TURNS: _w28('F7', 'P2: an owner whose Psi turns more than allowed per cycle is refused (A1)'),
    Rule.SECOND_CAPTURE_F3: _w28('owner', 'second capture excluded: x follows the secular harmonics, not sin x (A2)'),
    Rule.BOTH_RESONANCES_CONFIRMED: _w28('2j', "'both resonances at once?' answered confirmed, origin joint (A2b)"),
    Rule.FRAME_JOIN: _w28('G2q', 'episodes joined over a psi-frame half-cycle (rule q, B1)'),
    Rule.STALL_JOIN: _w28('G2s', 'a stall remainder joined to its episode (rule s, B1)'),
    Rule.REGIME_CUT_MARK: _w28('G2m', 'a centre step or stall inside one libration region kept as a mark, not a boundary (C16)'),
    Rule.BOUNDED_GAP_MERGE: _w28('G2u', 'episodes joined: sigma stays inside the band through the gap (A7)'),
    Rule.LOCAL_CENTRE_ENTRY: _w28('G2', 'entry or exit read against the local centre, other than the mean centre (C9)'),
    Rule.FORCED_STRENGTH_MARK: _w28('5s', 'the episodes of a 5s row marked forced (strength) (B2)'),
    Rule.O1_FOREIGN_WAVE: _w28('O1', 'O1 objects to a coherent foreign a-wave only (B4)'),
    Rule.R0_WINDOWS: _w28('R0', 'R0 counts only windows where sigma swings enough (B3)'),
    Rule.POTENTIAL_ON_ARC: _w28('R0', 'R0 compares V_F on the visited arc, without a free constant (B3)'),
    Rule.SUBFLOOR_PASS: _w28('G1f', 'G1f: a two-body cycle below the period floor passed (B5)'),
    Rule.ORBIT_CUTOFF: _w28('Pc', 'the cutoff raised to the body orbital period bound (B6a)'),
    Rule.RIPPLE_ON_ARC: _w28('F6r', 'a ripple without physics on an unfinished slow swing does not decide (B6b)'),
    Rule.SLOW_SWING_IN_WELL: _w28('G5w', 'G5w: an unfinished slow swing compatible with the well (B6c)'),
    Rule.SCALE_LADDER: _w28('G2c', 'slow libration found at a shorter cutoff of the scale ladder (B7)'),
    Rule.DIRECT_FORCE_LAW: _w28('L4d', 'the direct averaged-force law at the libration scale decided L4 (B8)'),
    Rule.EXTEND_LENGTH_LIMITED: _w28('extend', 'a length-limited uncertain row given extend (B9)'),
    Rule.EQ_SCALE_NOT_OBJECTION: _w28('5x', 'leaf 5x: an eq_scale strength failure is no objection (B9)'),
    Rule.SLOW_SCALE: _w28('G2c', 'the slow component stored and judged at its own scale (B10a)'),
    Rule.CENTRE_SCALE_DISPLACED: RuleInfo(
        'G2c', 'a center-scale reading with a measured objection does not displace a fine dynamic libration (709123)', 'v1.1.0'
    ),
    Rule.SECULAR_CYCLE_KINEMATIC: _w28('K3s', 'K3s: the slow cycle is a secular cycle, kinematic libration (B10a)'),
    Rule.KINEMATIC_HIGH: _w28('K1e', 'a kinematic libration with measured physics allowed to reach high (A5)'),
    Rule.SUBRESONANCE_THEORY_SELECT: _w28('S', 'the theory chose the subresonance (A6b)'),
    Rule.SUBRESONANCE_T7: _w28('S', 'subresonance scan triggered by T7 (A6a)'),
    Rule.SUBRESONANCE_T8: _w28('S', 'subresonance scan triggered by T8 (A6a)'),
    Rule.COORBITAL_FLOOR_PROFILE: _w28('G2', "1:1 period floor from the resonance's own profile (C3)"),
    Rule.COORBITAL_QS_RUN_FLOOR: RuleInfo('G1q', "1:1 quasi-satellite run's floor from the loop through its own turns (G1q pass)", 'W38'),
    Rule.COORBITAL_PAIR_LOOP_FLOOR: RuleInfo(
        'G2', "1:1 run with a non-reflecting turn: its floor from its reflecting pairs' loop, its own force law passing", 'W38'
    ),
    Rule.COORBITAL_EDGE_TURN: _w28('G2', '1:1 edge turning point from the orbit-averaged raw angle closed a cycle (C4)'),
    Rule.QS_TRIGGER: _w28('G1q', 'G1q quasi-satellite pass triggered at the lower visit length (C5)'),
    Rule.COORBITAL_LEG: _w28('G4c', '1:1 single-leg branch G4c decided (C6)'),
    Rule.COORBITAL_NEAR_EPISODE: _w28('G4c', '1:1 structural near-episode cap (C6)'),
    Rule.COORBITAL_WELL_CUT: _w28('G2', '1:1 well or amplitude change kept as an annotation, not a boundary (C7)'),
    Rule.COORBITAL_WELL_CORE: RuleInfo(
        'G2', "1:1 run kept from its well-holding core, continued to the next separatrix crossing (JR16)", 'W38'
    ),
    Rule.COORBITAL_RESIDENCE: _w28('G2', '1:1 core, return, residence and sub-regimes stored (C8)'),
    Rule.SINGLE_HALF_LIBRATION: _w28('G4h', 'a half-swing of another MMR between passages with its own law: uncertain (C6, q-5 i)'),
    Rule.ELIGIBLE_REASON: _w28('G2', 'an accepted episode is ineligible, with its reason (C11)'),
    Rule.END_ARC_EXTEND: _w28('end_arc_extend', 'a record ending in an arc after circulation: extend_time, record_ends_unresolved (C14)'),
    Rule.FREE_RESAMPLE_REMAINDER: _w28('FE', 'free elements: anti-alias filter at a decimation with a remainder (D1)'),
    Rule.ZLK_INVARIABLE_FRAME: _w28('ZF', 'ZLK omega measured in the invariable plane (D2)'),
    Rule.ZLK_HALF_TURN_CUT: _w28('G2z', 'ZLK run broken at half a turn (D3)'),
    Rule.ZLK_COORBITAL_EPISODIC: _w28('SZ.G', 'a bounded, non-librating ZLK omega of an unfitted co-orbital sent to -9 (D4)'),
    Rule.CENTRE_TURNED_INSUFFICIENT: _w28('G5s', 'a centre that turned back reads insufficient_time, not drift (D6)'),
    Rule.PHASE_UNRESOLVED: _w28('G5p', 'a secular swing within the uncertainty of the mode phases (D5)'),
    Rule.ZLK_GATE_UNCONFIRMED: _w28('SZ.ei', 'episodic ZLK without a gate verdict: status by the episode e-i coupling (D10)'),
}

_ORDER = {rule: index for index, rule in enumerate(RULES)}


def rules_since(tag: str) -> frozenset[Rule]:
    """The rules introduced in iteration `tag` ('W28')."""
    return frozenset(rule for rule, info in RULES.items() if info.since == tag)


def fire(fired: Iterable[Rule | str], *rules: Rule | str) -> frozenset[Rule]:
    """`fired` with `rules` added (a rule fires once however often its owner calls this); a name outside the registry
    raises ValueError. Owners write the result back, e.g. `geometry.rules_fired = fire(geometry.rules_fired, Rule.X)`."""
    return frozenset(Rule(rule) for rule in fired) | {Rule(rule) for rule in rules}


def ordered(*groups: Iterable[Rule | str]) -> list[str]:
    """The rules of every group, deduplicated, in registry order (`Verdict.rules_fired`, the decide steps' merge). A
    name outside the registry (a file of another version read back) is kept, after the registered ones, in the order
    first seen."""
    names = list(dict.fromkeys(str(rule) for group in groups for rule in group))
    known = sorted((name for name in names if name in _ORDER), key=lambda name: _ORDER[Rule(name)])
    return known + [name for name in names if name not in _ORDER]
