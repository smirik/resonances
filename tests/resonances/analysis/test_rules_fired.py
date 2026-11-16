"""W28 F1, rule flags (cards s-4, g10-3): the registry of named rules, `fire`, the merge into `Verdict.rules_fired` and
the summary column `rules_fired`.

The expected sets and orders are written out here from the work package's list of pre-registered names, not read back
from the registry. A row on which no W28 rule acted (a clean pendulum libration, a secular libration) has an empty
column; the figure never prints it.
"""

import dataclasses
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from resonances.plotting import combined_numbers
from resonances.resonance.analysis import rules
from resonances.resonance.analysis.mmr import decide as mmr_decide
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.rules import RULES, Rule, fire, ordered, rules_since
from resonances.resonance.analysis.secular import LongPeriodSeries, analyse_secular
from resonances.resonance.analysis.secular import decide as secular_decide
from resonances.resonance.analysis.settings import AnalysisSettings, settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, Reason, ResonanceType, Status, Verdict
from tests.resonances.analysis.builders import make, make_geometry
from tests.resonances.analysis.test_impl_w3_guards_input import pendulum_series

S = settings_from_config()

# The work package's list, by area (since = 'W28')
W28_NAMES = {
    'owner_per_episode',
    'owner_psi_turns',
    'second_capture_f3',
    'both_resonances_confirmed',
    'frame_join',
    'stall_join',
    'regime_cut_mark',
    'bounded_gap_merge',
    'local_centre_entry',
    'forced_strength_mark',
    'o1_foreign_wave',
    'r0_windows',
    'potential_on_arc',
    'subfloor_pass',
    'orbit_cutoff',
    'ripple_on_arc',
    'slow_swing_in_well',
    'scale_ladder',
    'direct_force_law',
    'extend_length_limited',
    'eq_scale_not_objection',
    'slow_scale',
    'secular_cycle_kinematic',
    'kinematic_high',
    'subresonance_theory_select',
    'subresonance_t7',
    'subresonance_t8',
    'coorbital_floor_profile',
    'coorbital_edge_turn',
    'qs_trigger',
    'coorbital_leg',
    'coorbital_near_episode',
    'coorbital_well_cut',
    'coorbital_residence',
    'single_half_libration',  # W28 C6, q-5 i (added after the plan's list)
    'eligible_reason',
    'end_arc_extend',
    'free_resample_remainder',
    'zlk_invariable_frame',
    'zlk_half_turn_cut',
    'zlk_coorbital_episodic',
    'centre_turned_insufficient',
    'phase_unresolved',
    'zlk_gate_unconfirmed',
}


# --- the registry ------------------------------------------------------------------------------------------------------


def test_every_rule_has_one_rule_info():
    assert list(RULES) == list(Rule)  # one entry each, in the enum's order (the registry order)
    for rule, info in RULES.items():
        assert info.title and info.since and info.doc_id, rule


def test_the_w28_rules_are_the_pre_registered_names():
    assert {str(rule) for rule in rules_since('W28')} == W28_NAMES
    assert len(W28_NAMES) == 44
    assert rules_since('W99') == frozenset()


# The rows of docs/libration-analysis.md the work packages named for their rules (their reports; E16 wrote them in)
W28_DOC_IDS = {
    'owner_per_episode': 'F7',
    'owner_psi_turns': 'F7',
    'second_capture_f3': 'owner',
    'both_resonances_confirmed': '2j',
    'frame_join': 'G2q',
    'stall_join': 'G2s',
    'regime_cut_mark': 'G2m',
    'bounded_gap_merge': 'G2u',
    'local_centre_entry': 'G2',
    'forced_strength_mark': '5s',
    'o1_foreign_wave': 'O1',
    'r0_windows': 'R0',
    'potential_on_arc': 'R0',
    'subfloor_pass': 'G1f',
    'orbit_cutoff': 'Pc',
    'ripple_on_arc': 'F6r',
    'slow_swing_in_well': 'G5w',
    'scale_ladder': 'G2c',
    'direct_force_law': 'L4d',
    'extend_length_limited': 'extend',
    'eq_scale_not_objection': '5x',
    'slow_scale': 'G2c',
    'secular_cycle_kinematic': 'K3s',
    'kinematic_high': 'K1e',
    'subresonance_theory_select': 'S',
    'subresonance_t7': 'S',
    'subresonance_t8': 'S',
    'coorbital_floor_profile': 'G2',
    'coorbital_edge_turn': 'G2',
    'qs_trigger': 'G1q',
    'coorbital_leg': 'G4c',
    'coorbital_near_episode': 'G4c',
    'coorbital_well_cut': 'G2',
    'coorbital_residence': 'G2',
    'single_half_libration': 'G4h',
    'eligible_reason': 'G2',
    'end_arc_extend': 'end_arc_extend',
    'free_resample_remainder': 'FE',
    'zlk_invariable_frame': 'ZF',
    'zlk_half_turn_cut': 'G2z',
    'zlk_coorbital_episodic': 'SZ.G',
    'centre_turned_insufficient': 'G5s',
    'phase_unresolved': 'G5p',
    'zlk_gate_unconfirmed': 'SZ.ei',
}

RULE_INDEX = Path(__file__).resolve().parents[3] / 'docs' / 'libration-analysis.md'


def rule_index_rows() -> list[tuple[set[str], str]]:
    """Every table row of the rule index: the ids of its first cell ('R0, R0σ', 'S (scan)', '5s, 5s\\*') and the row."""
    rows = []
    for line in RULE_INDEX.read_text().splitlines():
        if not line.startswith('| ') or line.startswith('|---'):
            continue
        first = line.split(' | ')[0].lstrip('| ').replace('\\', '')
        rows.append(({token for token in re.split(r'[,\s]+', first) if token}, line))
    return rows


def test_every_w28_rule_has_the_doc_id_of_its_row():
    assert set(W28_DOC_IDS) == W28_NAMES
    assert {str(rule): info.doc_id for rule, info in RULES.items() if rule in rules_since('W28')} == W28_DOC_IDS
    assert rules.PENDING not in {info.doc_id for info in RULES.values()}


def test_every_w28_rule_is_named_in_a_row_of_its_doc_id():
    # the rule index has a row with that id, and one such row names the rule ('end_arc_extend' is its own id)
    rows = rule_index_rows()
    for name, doc_id in W28_DOC_IDS.items():
        own = [line for ids, line in rows if doc_id in ids]
        assert own, (name, doc_id)
        assert any(f'`{name}`' in line or line.startswith(f'| {name} |') for line in own), (name, doc_id)


# --- fire and the merge ------------------------------------------------------------------------------------------------


def test_fire_deduplicates():
    fired = fire(frozenset(), Rule.FRAME_JOIN)
    fired = fire(fired, Rule.FRAME_JOIN, 'frame_join', Rule.OWNER_PER_EPISODE)
    assert fired == frozenset({Rule.FRAME_JOIN, Rule.OWNER_PER_EPISODE})
    assert fire(fired) == fired  # nothing added, nothing lost


def test_fire_refuses_a_name_outside_the_registry():
    with pytest.raises(ValueError):
        fire(frozenset(), 'no_such_rule')


def test_the_merge_keeps_registry_order_whatever_the_firing_order():
    # registry order: owner_per_episode (owner area) < frame_join (joins) < zlk_half_turn_cut (secular/ZLK)
    late_first = fire(fire(frozenset(), Rule.ZLK_HALF_TURN_CUT), Rule.FRAME_JOIN)
    assert ordered(late_first, [Rule.OWNER_PER_EPISODE, Rule.FRAME_JOIN]) == ['owner_per_episode', 'frame_join', 'zlk_half_turn_cut']


def test_the_merge_keeps_an_unknown_name_after_the_registered_ones():
    # a summary.csv of another version read back: its names survive, after the registry's
    assert ordered(['old_rule', 'stall_join', 'frame_join']) == ['frame_join', 'stall_join', 'old_rule']


def test_mmr_merge_reads_the_geometry_and_the_other_results():
    geometry = make_geometry(rules_fired=fire(frozenset(), Rule.STALL_JOIN))
    assert mmr_decide.rules_fired(geometry, frozenset({Rule.OWNER_PSI_TURNS})) == ['owner_psi_turns', 'stall_join']
    assert mmr_decide.rules_fired(None) == []  # a verdict before the measurements


def test_secular_merge_reads_geometry_angle_status_and_the_gate():
    geometry = make_geometry(rules_fired=frozenset({Rule.ZLK_HALF_TURN_CUT}))
    angle = secular_decide.AngleDecision(AngleStatus.EPISODIC, rules_fired=frozenset({Rule.CENTRE_TURNED_INSUFFICIENT}))
    status = secular_decide.StatusDecision(Status.UNCERTAIN, Reason.GATE_UNAVAILABLE, rules_fired=frozenset({Rule.ZLK_GATE_UNCONFIRMED}))
    merged = secular_decide.rules_fired(geometry, angle, status, frozenset({Rule.ZLK_HALF_TURN_CUT, Rule.ZLK_INVARIABLE_FRAME}))
    assert merged == ['zlk_invariable_frame', 'zlk_half_turn_cut', 'centre_turned_insufficient', 'zlk_gate_unconfirmed']


def test_builders_fill_a_frozenset_field():
    @dataclasses.dataclass
    class Result:
        rules_fired: frozenset

    assert make(Result).rules_fired == frozenset()


# --- the column --------------------------------------------------------------------------------------------------------


def two_rule_verdict() -> Verdict:
    # fired in reverse registry order: the column still lists them in registry order
    return Verdict(ResonanceType.MMR, Status.TRANSIENT, AngleStatus.EPISODIC, rules_fired=['bounded_gap_merge', 'owner_per_episode'])


def test_a_verdict_with_two_rules_round_trips_through_the_flat_dict():
    row = two_rule_verdict().to_flat_dict()
    assert row['rules_fired'] == 'owner_per_episode;bounded_gap_merge'
    back = Verdict.from_flat_dict(row)
    assert back.rules_fired == ['owner_per_episode', 'bounded_gap_merge']
    assert back.to_flat_dict() == row


def test_the_column_survives_a_csv_and_an_empty_cell_is_no_rule(tmp_path):
    path = tmp_path / 'summary.csv'
    pd.DataFrame(
        [two_rule_verdict().to_flat_dict(), Verdict(ResonanceType.SECULAR, Status.RESONANT, AngleStatus.LIBRATION).to_flat_dict()]
    ).to_csv(path, index=False)
    rows = pd.read_csv(path).to_dict('records')
    assert Verdict.from_flat_dict(rows[0]).rules_fired == ['owner_per_episode', 'bounded_gap_merge']
    assert Verdict.from_flat_dict(rows[1]).rules_fired == []  # the empty cell reads back as NaN


def test_the_flags_are_no_setting_and_no_figure_line():
    assert 'rules_fired' not in {field.name for field in dataclasses.fields(AnalysisSettings)}
    lines = combined_numbers.other_lines({'rules_fired': 'frame_join', 'comments': ''}, {}, {}, {})
    assert [line.column for line in lines] == ['comments']


# --- rows on which no W28 rule acted ------------------------------------------------------------------------------------


def test_a_clean_synthetic_pendulum_fires_no_rule():
    analysis = analyse_mmr(pendulum_series(), S, scan_owner=False, scan_subresonances=False)
    assert analysis.verdict.status == Status.RESONANT  # the clean case: a libration over the record
    assert analysis.verdict.rules_fired == []
    assert analysis.summary_row()['rules_fired'] == ''


def test_a_clean_secular_libration_fires_no_rule():
    # 10 Myr at 500 yr, a 1-Myr libration of 60 deg about 180 deg (a centre scale is not needed)
    t = np.linspace(0.0, 1e7, 20001)
    angle = np.pi + np.radians(60.0) * np.sin(2 * np.pi * t / 1e6)
    n = len(t)
    series = LongPeriodSeries(ResonanceType.SECULAR, 'test', 'synthetic', t, angle, np.full(n, 2.5), np.full(n, 0.1), np.full(n, 0.1))
    analysis = analyse_secular(series, S)
    assert analysis.verdict.status == Status.RESONANT
    assert analysis.verdict.rules_fired == []
    assert analysis.verdict.to_flat_dict()['rules_fired'] == ''
