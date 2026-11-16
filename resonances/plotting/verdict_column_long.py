"""The right column of a secular or Lidov-Kozai figure as a `verdict_column.Column` (decisions D1-D10, 2026-11-04).

Every block but Why (`verdict_column_why.long_why`) is built here from the predicates and clause functions of
`verdict_text` (the gate's outcome, the angle's shape, the e-i exchange, the time scales) and from the summary row; no
number is computed twice: every printed number is a ledger value, a CSV column or a FREQUENCIES row of the figure.

The steps of a long-period row (the ◆ on the one that set the status; `pol_decided_by` names the policy's step):

    Q   Can the data show it?          the filter guard (SZ.F); an unphysical orbit stops here
    0   Does σ (ω) librate?             the shape: turning points (SZ.N), center stationary (SZ.S), a circulating center
                                        (SZ.P), a center that turned back (G5s); ZLK: the center scale as a sub-check
    1   secular: the center scale       whether the slower center scale decided (`sec_scale` center)
        ZLK: Does the free ω librate?   the free-ω gate (SZ.G): its cycles, the free e against the forced e, the
                                        forcing's beat, the node not separated from s_j, a co-orbital it cannot fit
    2   secular: the phase error        the swing against the error of the linear mode phases (G5p)
        ZLK: the e-i exchange           the deciding episode's e and i at constant √(1−e²) cos i (read by the level,
                                        a veto zlk_integral, or flag kinematic_excluded)
    P   the policy's line               ZLK: a gate without a verdict keeps the shape's status; any row the policy
                                        decided as too short (`pol_decided_by` insufficient_time): the end travel of σ_f

Words for the long-period inputs of the level replace the MMR ones of `verdict_column_common` (`LONG_PILL_WORDS`)."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from resonances.plotting import verdict_column as vc
from resonances.plotting import verdict_column_common as common
from resonances.plotting import verdict_column_mmr as mmr
from resonances.plotting import verdict_column_why as why
from resonances.plotting import verdict_text as vt
from resonances.plotting import verdict_vocab as vocab
from resonances.resonance.analysis import policy as pol
from resonances.resonance.analysis.secular.decide import EDGE_ARC_ENTRIES
from resonances.resonance.analysis.secular.models import ARCSEC_PER_TURN
from resonances.resonance.analysis.zlk.episode_ei import phase_off_deg

DOCS = 'physics/secular-zlk.md'
NOUN = {'secular': 'σ', 'zlk': 'ω'}
# what set the filter cutoff P_c (`secular.preprocess.choose_cutoff`: the largest of a multiple of the great
# inequality's period, of the slowest forced wave's and the type's shortest libration period over the guard's factor)
BINDING_WORDS = {
    'great_inequality': 'the Jupiter–Saturn great inequality',
    'forced_wave': 'the slowest forced wave',
    'shortest_libration': 'the shortest libration period accepted',
}
BINDING_SHORT = {'great_inequality': 'great inequality', 'forced_wave': 'slowest forced wave', 'shortest_libration': 'shortest libration'}
# the margins of a long-period row under their step (`_long_period_margin_row` words; SZ.G under the gate)
MARGIN_STEP = {'F': 'Q', 'N': '0', 'S': '0', 'P': '0', 'G': '1'}
STEP_WORDS = {
    'Q': 'data',
    '0': 'shape',
    'secular:1': 'center scale',
    'secular:2': 'phase error',
    'zlk:1': 'free-ω gate',
    'zlk:2': 'e-i exchange',
}
# the policy's line when the policy, not the tree, set the status (`pol_decided_by` gate_unavailable): a ZLK row whose
# free-ω gate gives no verdict keeps the shape's status, at most medium unless a kinematic libration is excluded
POLICY_NUMBER = 'P'
POLICY_GATE_QUESTION = 'Does the free-ω gate give a verdict?'
# the policy's line of a row it decided as too short (`policy._insufficient`, MMR's question): the end travel of σ_f
POLICY_SHORT_QUESTION = mmr.POLICY_QUESTIONS[pol.DECIDED_INSUFFICIENT][0]
OVERRULED = f'the policy decides at step {POLICY_NUMBER}'
SHORT_DECISIVE = {
    'non_resonant': 'Too few turning points for one cycle, and σ_f spans at least one turn to the record’s end: the record '
    'ends in circulation, decided now.',
    '': 'Too few turning points for one cycle: the record is too short to decide.',
}
# the gate's cycle count: the record over the libration period, not the cycles counted nor the free ω's turns
GATE_CYCLES_WORDS = 'cycles of ω in the record T / P_cyc'
# the policy's veto comparisons of the free ω (ledger names after 'policy:veto zlk_…: ') in words
VETO_ENTRY_WORDS = {
    'gate cycles': GATE_CYCLES_WORDS,
    'rho_max < 1': 'free e / forced e over the record ρ_max',
    'rho_episodes < 1': 'free e / forced e in the episodes ρ_episodes',
    'free omega turns per cycle': 'turns of the free ω per cycle',
}
EPISODES_SHOWN = 4  # librating episodes listed in Key facts (the deciding one always)


# --------------------------------------------------------------------------------------------- small helpers


def _signed(value: Any) -> str:
    return vt.fmt(value).replace('-', '−')


def _deg(value: Any) -> str:
    return f'{vt.fmt(value)}°'


def _turns(entry) -> str:
    """A ledger comparison in turns ('121 turns ≥ 1 turn'); the Numbers block keeps 'rev'."""
    return re.sub(r'(?<![\d.])1 turns\b', '1 turn', _entry_text(entry, ' turns'))


def resonance_words(name: str) -> str:
    """'g-g6+s-s6' -> 'g − g6 + s − s6'; '2_g-g6_+_s-s6_' -> '2(g − g6) + (s − s6)'; 'LK' -> 'Lidov–Kozai'."""
    name = common.text(name)
    if name in ('', 'LK'):
        return 'Lidov–Kozai'
    if '_' in name:  # the catalogue's grouped form: 2_g-g6_+_s-s6_
        name = name.replace('_+_', ')+(').replace('_-_', ')-(').replace('_', '(', 1).rstrip('_') + ')'
    out = ''
    for i, ch in enumerate(name):
        if ch in '+-' and i > 0 and name[i - 1] not in '(':
            out += f' {"−" if ch == "-" else "+"} '
        else:
            out += '−' if ch == '-' else ch
    return out


def _gate(analysis):
    return getattr(analysis, 'gate', None)


def _outcome(analysis) -> str:
    gate = _gate(analysis)
    return common.text(gate.outcome) if gate is not None else ''


def _decided_by(row: Mapping[str, Any]) -> str:
    return common.text(row.get('pol_decided_by')).split(';')[0]


def _veto(analysis) -> str:
    return common.text(getattr(analysis.verdict, 'veto', ''))


def _entries(analysis) -> Dict[str, Any]:
    """The decision path by entry name (the last entry of a name wins)."""
    return {entry.name: entry for entry in analysis.verdict.decision_path}


def _entry_text(entry, unit: str = '') -> str:
    return vt.entry_compare(entry, unit) if entry is not None and vt.finite(entry.value) else ''


def _eps(analysis, episodes: Sequence[Mapping[str, Any]]) -> vt.Episodes:
    rows = vt.librating_episode_rows(analysis, episodes)
    return vt.Episodes(rows, vt.record_span_yr(analysis), analysis.decisive_episode_id)


def _unphysical(analysis) -> bool:
    v = analysis.verdict
    return str(v.status) == 'invalid' and str(v.reason) == 'unphysical'


# --------------------------------------------------------------------------------------------- the deciding step


def deciding_step(analysis, row: Mapping[str, Any], family: str) -> str:
    """The step that set the status ('Q', '0', '1', '2', 'P'): an unphysical orbit Q; a row the policy decided as too
    short (`pol_decided_by` insufficient_time: the record ends in circulation, W36 F5) the policy's line P; a ZLK veto
    of the free ω, a confirmed or kinematic gate 1 and a veto of the e-i integral 2; a gate without a verdict, on which
    the policy keeps the shape's status (`pol_decided_by` gate_unavailable), the policy's line P; secular: the phase
    error 2, the center scale 1; else the shape 0."""
    v = analysis.verdict
    if _unphysical(analysis):
        return 'Q'
    by = _decided_by(row)
    if by == pol.DECIDED_INSUFFICIENT:
        return POLICY_NUMBER
    if family == 'zlk':
        if by == pol.DECIDED_GATE_UNAVAILABLE:
            return POLICY_NUMBER
        if by.startswith(pol.DECIDED_VETO):
            return '2' if by == pol.DECIDED_VETO + pol.VETO_ZLK_INTEGRAL else '1'
        if by == pol.DECIDED_GATE_CONFIRMED:
            return '1'
        if by == pol.DECIDED_TREE and _outcome(analysis) in vt.GATE_VERDICTS and str(v.angle_status) in ('libration', 'episodic'):
            return '2' if _veto(analysis) == pol.VETO_ZLK_INTEGRAL else '1'
        return '0'
    if str(v.angle_subtype) == 'phase_unresolved':
        return '2'
    if common.text(row.get('sec_scale')) == 'centre':
        return '1'
    return '0'


# --------------------------------------------------------------------------------------------- checks


ICON_OF_STATE = {'pass': vc.OK, 'thin': vc.THIN, 'fail': vc.FAIL, 'not_applicable': vc.QUIET}
# why a margin does not apply to a long-period row (`secular.decide`: S only for an episode whose centre is modulated,
# a bounded centre that turns back LIBRATION_CENTRE_RETURNS_MODULATED times or more)
NOT_APPLICABLE_WHY = {
    'S': 'not applicable: a modulated center, it swings back and forth within a bounded range',
}


def _margin_sub(key: str, margin, settings, whole: bool) -> vc.SubCheck:
    """A margin as a sub-check: its name, the comparison as it came out (and what a full margin needs), its id."""
    check = vt._long_period_margin_row(key, margin, settings, whole)
    icon = ICON_OF_STATE.get(str(getattr(check.state, 'value', check.state)), vc.MISSING)
    if icon == vc.QUIET:
        return vc.SubCheck(icon, check.name, check.note or NOT_APPLICABLE_WHY.get(key, 'not applicable'), check.rule_id)
    need = f'need {check.need}' if check.need and icon != vc.OK else ''
    numbers = '; '.join(part for part in (check.detail, need, check.note if icon != vc.OK else '') if part)
    return vc.SubCheck(icon, check.name, numbers, check.rule_id)


def _margin_subs(analysis, settings, step: str) -> List[vc.SubCheck]:
    v = analysis.verdict
    whole = str(v.angle_status) == 'libration'
    return [_margin_sub(key, m, settings, whole) for key, m in v.margins.items() if MARGIN_STEP.get(key) == step]


def _line(number, question, answer='', numbers='', rule_id='', children=(), decides=False, quiet=False, icon=None) -> vc.CheckLine:
    if decides:
        icon = vc.DECIDES
    elif icon is None:
        icon = common.worst(children)
    return vc.CheckLine(icon, number, question, answer, numbers, rule_id, tuple(children), decides, quiet or icon == vc.QUIET)


def _q_step(analysis, row, settings, decides: bool) -> vc.CheckLine:
    if _unphysical(analysis):
        words = vt.unphysical_text(analysis, row) or 'the orbit is unphysical'
        return _line('Q', mmr.QUALITY_QUESTION, f'No: {words}; no angle is read.', decides=decides)
    children = _margin_subs(analysis, settings, 'Q')
    span, step, cutoff = vt._float(row.get('smp_span_yr')), vt._float(row.get('smp_step_yr')), vt._float(row.get('flt_cutoff_yr'))
    numbers = ', '.join(
        part
        for part in (
            f'T {vt.years(span)}' if vt.finite(span) else '',
            f'step {vt.years(step)}' if vt.finite(step) else '',
            f'cutoff P_c {vt.years(cutoff)}' if vt.finite(cutoff) else '',
        )
        if part
    )
    icon = common.worst(children)
    answer = {vc.OK: 'Yes.', vc.THIN: 'Yes, barely.', vc.MISSING: 'Not sure.', vc.FAIL: 'No.'}[icon]
    return _line('Q', mmr.QUALITY_QUESTION, answer, numbers, children=children, decides=decides)


def _shape_answer(analysis, row, eps: vt.Episodes, family: str) -> str:
    v = analysis.verdict
    angle, subtype = str(v.angle_status), str(v.angle_subtype)
    rules = set(v.rules_fired)
    if angle == 'libration':
        return 'Yes, over the whole record.'
    if angle == 'episodic':
        n = len(eps.rows)
        return f'In part of the record: {vt.plural(n, "librating episode")}.'
    if angle == 'circulation':
        return 'No, it circulates.'
    if angle == 'drift':
        return 'No: it oscillates about a center that moves one way (a drift).'
    if angle == 'near_separatrix':
        return 'No: it slips over the top.'
    if angle == 'insufficient_time':
        if subtype == 'phase_unresolved':
            return 'It swings, within the error of the mode phases.'
        if 'centre_turned_insufficient' in rules:
            return 'Not yet: the center turned back within less than a turn (part of one slow swing).'
        if subtype in ('arc', 'stall'):
            return f'Not yet: no turning point, less than a turn ({subtype}).'
        return 'Not yet: fewer turning points than one cycle needs.'
    return f'{vocab.snake_words(angle)}.'


def _shape_numbers(analysis) -> str:
    """The comparisons of the shape as the ledger made them (the episodes, the coverage, a circulating center)."""
    entries = _entries(analysis)
    v = analysis.verdict
    angle = str(v.angle_status)
    parts = []
    count = entries.get('librating_episodes')
    if count is not None and vt.finite(count.value) and angle in ('libration', 'episodic'):
        parts.append(f'librating episodes {_entry_text(count)}')
    start, end = entries.get('coverage_start'), entries.get('coverage_end')
    if start is not None and end is not None and vt.finite(start.value) and vt.finite(end.value):
        parts.append(
            f'gaps to the record’s ends {vt.fmt(start.value)} P and {vt.fmt(end.value)} P (a gap of ≤ {vt.fmt(start.threshold)} '
            f'libration period{"" if start.threshold == 1 else "s"} counts as reaching the end)'
        )
    for kind, where in (('entry', 'before the first turning point'), ('exit', 'after the last turning point')):
        arc = entries.get(EDGE_ARC_ENTRIES[kind])
        if arc is not None and vt.finite(arc.value):
            parts.append(f'{where} σ_f runs {_turns(arc)} one way: the {kind} is seen, that end is not reached')
    passed = entries.get('centre_pass')
    if passed is not None and vt.finite(passed.value) and angle in ('circulation', 'drift', 'insufficient_time'):
        parts.append(f'center’s largest one-way pass {_turns(passed)}')
    drift = entries.get('drift_fraction')
    if drift is not None and vt.finite(drift.value) and angle in ('drift', 'insufficient_time'):
        parts.append(f'moving-center runs cover {vt.compare_percent(drift.value, ">=", drift.threshold)} of the window')
    return '; '.join(parts)


def _centre_scale_sub(analysis, row) -> Optional[vc.SubCheck]:
    """The center scale as one sub-check (ZLK): what it found when it ran, else None."""
    entries = {name.split(':', 1)[1]: entry for name, entry in _entries(analysis).items() if name.startswith('centre_scale:')}
    if not entries:
        return None
    if common.text(row.get('sec_scale')) == 'centre':
        return vc.SubCheck(vc.OK, 'the center scale decides: its center librates', _centre_scale_numbers(entries))
    return vc.SubCheck(vc.QUIET, 'center scale: no slower libration', _centre_scale_numbers(entries))


def _centre_scale_numbers(entries: Mapping[str, Any]) -> str:
    parts = []
    if 'librating_episodes' in entries and vt.finite(entries['librating_episodes'].value):
        parts.append(f'episodes {_entry_text(entries["librating_episodes"])}')
    if 'centre_pass' in entries and vt.finite(entries['centre_pass'].value):
        parts.append(f'center pass {_turns(entries["centre_pass"])}')
    return '; '.join(parts)


def _shape_step(analysis, row, settings, eps, family, decides: bool) -> vc.CheckLine:
    v = analysis.verdict
    children = _margin_subs(analysis, settings, '0')
    entries = _entries(analysis)
    turned = entries.get('centre_monotone')
    if 'centre_turned_insufficient' in set(v.rules_fired):
        numbers = f'interior turns of the slow center {vt.fmt(turned.value)}' if turned is not None and vt.finite(turned.value) else ''
        children.append(vc.SubCheck(vc.MISSING, 'the slow center turned back: an unfinished swing, not a drift', numbers))
    if family == 'zlk':
        sub = _centre_scale_sub(analysis, row)
        if sub is not None:
            children.append(sub)
    question = f'Does {NOUN[family]} librate?'
    icon = None if children else vc.OK
    rule = 'G5s' if 'centre_turned_insufficient' in set(v.rules_fired) else ''
    answer = _shape_answer(analysis, row, eps, family)
    if _decided_by(row) == pol.DECIDED_INSUFFICIENT:  # the shape's 'not yet' is the tree's; the policy's line decides
        answer = f'{answer.rstrip(".")}; {OVERRULED}.'
    return _line('0', question, answer, _shape_numbers(analysis), rule, children, decides, icon=icon)


def _secular_centre_step(analysis, row, decides: bool) -> Optional[vc.CheckLine]:
    entries = {name.split(':', 1)[1]: entry for name, entry in _entries(analysis).items() if name.startswith('centre_scale:')}
    question = 'Does its center librate at the slower center scale?'
    if not entries and str(analysis.verdict.angle_status) in ('libration', 'episodic'):
        return _line('1', question, 'Not needed: the fine scale shows the libration.', icon=vc.QUIET)
    if common.text(row.get('sec_scale')) == 'centre':
        numbers = _centre_scale_numbers(entries)
        cutoff = vt._float(row.get('sec_scale_cutoff_yr'))
        numbers = '; '.join(p for p in (f'cutoff {vt.years(cutoff)}' if vt.finite(cutoff) else '', numbers) if p)
        return _line('1', question, 'Yes: the center scale decides.', numbers, decides=decides, icon=vc.OK)
    if entries:
        return _line('1', question, 'No.', _centre_scale_numbers(entries), decides=decides, icon=vc.OK)
    return None


def _phase_step(analysis, row, decides: bool) -> Optional[vc.CheckLine]:
    """G5p: the swing against the error of the linear mode phases (None when it was not compared)."""
    entry = _entries(analysis).get('phase_uncertainty')
    swing, spread = vt._float(row.get('sec_phase_swing_deg')), vt._float(row.get('sec_phase_uncertainty_deg'))
    question = 'Is the swing larger than the error of the mode phases?'
    if entry is None and not decides:
        return None
    modes, largest = vt._phase_modes(analysis)
    names = ', '.join(modes) + (f' (mostly {largest})' if len(modes) > 1 and largest else '')
    numbers = (
        f'largest full swing {vt.compare(swing, ">", spread, "°")}, the phase error of {names or "the modes"}' if vt.finite(swing) else ''
    )
    if decides:
        return _line('2', question, 'No → its turning points may be those of the phases.', numbers, 'G5p', decides=True)
    held = vt.finite(entry.value) and vt.holds(entry.value, str(entry.op), entry.threshold)
    return _line('2', question, 'Yes.' if held else 'No.', numbers, 'G5p', icon=vc.OK if held else vc.FAIL)


def _gate_subs(analysis, row, settings) -> List[vc.SubCheck]:
    """The free-ω gate's sub-checks, each the way it came out."""
    gate, outcome = _gate(analysis), _outcome(analysis)
    children = _margin_subs(analysis, settings, '1')
    entries = _entries(analysis)
    if not any(child.rule_id == vocab.long_period_id('G') for child in children):
        cycles = entries.get('gate_cycles')
        if cycles is not None and not vt.finite(cycles.value):
            children.append(vc.SubCheck(vc.MISSING, 'cycles of ω for the gate', f'not measured (need ≥ {vt.fmt(cycles.threshold)})'))
    if outcome == 'confirmed':
        where = ' inside the librating episodes' if common.text(gate.detail) == 'episodes' else ''
        free = vt._float(row.get('free_omega_range_deg'))
        numbers = f'range of the free ω {_deg(free)}' if vt.finite(free) and free < 360 else ''
        children.append(vc.SubCheck(vc.OK, f'the free ω librates{where}', numbers))
    elif outcome == 'kinematic':
        rho = gate.rho_max
        if vt.finite(rho):
            children.append(
                vc.SubCheck(
                    vc.OK if rho < 1 else vc.QUIET,
                    # the name says what the numbers show (no – row states the passing condition beside failing ones)
                    'free e below the forced e over the record' if rho < 1 else 'free e reaches the forced e over the record',
                    f'ρ_max {vt.compare(rho, "<", 1.0)}, the largest local free/forced ratio along the record',
                )
            )
        if vt.finite(gate.rho_episodes):
            children.append(
                vc.SubCheck(
                    vc.OK if gate.rho_episodes < 1 else vc.QUIET,
                    'free e below the forced e in the episodes' if gate.rho_episodes < 1 else 'free e reaches the forced e in the episodes',
                    f'ρ_episodes {vt.compare(gate.rho_episodes, "<", 1.0)}, the largest local free/forced ratio inside them',
                )
            )
        zlk = getattr(analysis, 'zlk', None)
        attribution = zlk.attribution if zlk is not None else None
        if attribution is not None and attribution.attributed:
            mode = attribution.mode.split(' ')[0]
            children.append(
                vc.SubCheck(
                    vc.OK,
                    'the libration period is the free–forced beat',
                    f'2π/|b_ϖ − {mode}| {vt.years(attribution.beat_period_yr)}, P_cyc {vt.years(gate.libration_period_yr)}',
                )
            )
    elif outcome in vt.GATE_VERDICTS or not outcome or outcome == 'not_applicable':
        pass
    else:
        children.append(vc.SubCheck(vc.MISSING, *_no_verdict_parts(analysis, row)))
    return children


def _no_verdict_parts(analysis, row) -> Tuple[str, str]:
    """(what stopped the gate, its numbers) for a gate without a verdict."""
    gate, outcome = _gate(analysis), _outcome(analysis)
    if gate is not None and common.text(gate.detail) == 'coorbital_unfitted_forcing':
        return f'the free-element model does not apply to a co-orbital of {vt._coorbital_planet(row)}', ''
    mode = common.text(row.get('zlk_gate_node_near'))
    distance, resolution = vt._float(row.get('zlk_gate_node_near_d_arcsec_yr')), vt._float(
        row.get('zlk_gate_node_near_resolution_arcsec_yr')
    )
    if outcome == 'planetary_residual' and mode and vt.finite(distance):
        separation = vt._float(row.get('zlk_gate_node_separation_yr'))
        after = f'; they separate after {vt.years(separation)}' if vt.finite(separation) else ''
        return f'the free node is not yet separated from {mode}', f'|b_Ω − {mode}| {vt.compare(distance, "<", resolution, "″/yr")}{after}'
    if outcome == 'planetary_residual':
        return 'a residual rate of the free elements matches a planetary one', common.text(row.get('zlk_gate_matches')).replace(';', ', ')
    return why.GATE_NO_VERDICT[outcome][0] if outcome in why.GATE_NO_VERDICT else vocab.snake_words(outcome), ''


def _gate_step(analysis, row, settings, decides: bool, policy: bool = False) -> Optional[vc.CheckLine]:
    """Step 1 of a ZLK row, the free-ω gate; `policy`: the policy's line P carries what a gate without a verdict
    leaves to the shape, so step 1 only says that it gave none."""
    v, outcome = analysis.verdict, _outcome(analysis)
    if not outcome or outcome == 'not_applicable':
        return None
    children = _gate_subs(analysis, row, settings)
    entries = _entries(analysis)
    cycles = entries.get('gate_cycles')
    numbers = f'{GATE_CYCLES_WORDS} {_entry_text(cycles)}' if cycles is not None and vt.finite(cycles.value) else ''
    vetoes = [
        f'{VETO_ENTRY_WORDS.get(name.split(": ", 1)[-1], name.split(": ", 1)[-1])} {_entry_text(entry)}'
        for name, entry in entries.items()
        if name.startswith('policy:veto zlk_kinematic') or name.startswith('policy:veto zlk_free_e')
        if vt.finite(entry.value) and str(entry.mark) == '✓'
    ]
    if vetoes:
        numbers = '; '.join(vetoes)
    status = vt.status_head(v.status).lower()
    question = 'Does the free ω librate too?'
    if outcome == 'confirmed':
        answer = f'Yes → {status}, confirmed.' if decides else 'Yes (confirmed).'
        return _line('1', question, answer, numbers, 'SZ.G', children, decides)
    if outcome == 'kinematic':
        if decides:
            return _line('1', question, f'No, it winds → {status}, kinematic.', numbers, 'SZ.G', children, True)
        objection = pol.OBJECTION_ZLK_GATE in why.objection_checks(row)
        answer = 'It may wind: a measured objection; the shape keeps the status.' if objection else 'No, it winds.'
        return _line('1', question, answer, numbers, 'SZ.G', children, icon=vc.FAIL)
    if outcome == 'disabled':
        return _line('1', question, 'Not run: the gate is switched off.', '', 'SZ.G', children, icon=vc.MISSING)
    if decides and _veto(analysis) == pol.VETO_ZLK_FREE_E:
        # no verdict on the phase, but the policy's veto (`policy._zlk_measured`): the free e below the forced e over
        # the record with the e-i integral failed; `numbers` hold the veto's comparison
        answer = f'No verdict on the phase, but the free e stays below the forced e over the record → {status}, kinematic.'
        return _line('1', question, answer, numbers, 'SZ.G', children, True)
    if policy:
        answer = 'No verdict.'
    elif vt.has_mark(v, 'kinematic_excluded'):
        answer = 'No verdict; a kinematic libration is excluded (step 2) → the shape’s status is kept, at the level its evidence gives.'
    else:
        answer = 'No verdict → the shape’s status is kept, at most medium.'
    return _line('1', question, answer, numbers, 'SZ.G', children, decides, icon=vc.MISSING)


def _ei_read(analysis, row) -> bool:
    """Whether the status or the level reads the deciding episode's e-i exchange (`policy._long_outcome`): always where a
    gate applies, except under the veto zlk_kinematic and a kinematic gate's zlk_free_e over the record (ρ_max < 1),
    which need no e-i (a zlk_free_e inside the episodes or from an undefined phase needs the e-i failed too)."""
    gate, veto = _gate(analysis), _veto(analysis)
    if _outcome(analysis) in ('', 'not_applicable', 'disabled') or veto == pol.VETO_ZLK_KINEMATIC:
        return False
    if veto == pol.VETO_ZLK_FREE_E and _outcome(analysis) == 'kinematic':
        return not (vt.finite(gate.rho_max) and gate.rho_max < 1)
    return True


# the e-i sub-checks: (the name when the test holds, when it fails), so the name says what the numbers show
EI_NAMES = (
    ('i against e in antiphase', 'i against e not in antiphase'),
    ('e and i coherent', 'e and i not coherent'),
    ('√(1−e²) cos i kept', '√(1−e²) cos i not kept'),
)


def _ei_step(analysis, row, settings, decides: bool) -> Optional[vc.CheckLine]:
    """The e-i exchange of the deciding librating episode (`gate.episode_ei`), with the forced terms' size."""
    gate = _gate(analysis)
    ei = getattr(gate, 'episode_ei', None) if gate is not None else None
    question = 'Do e and i trade as a ZLK cycle in the deciding episode?'
    if ei is None or not _outcome(analysis) or _outcome(analysis) == 'not_applicable':
        return None
    read = _ei_read(analysis, row) or decides
    entries = _entries(analysis)
    small = entries.get('policy:zlk forced terms small: rho_max')
    children: List[vc.SubCheck] = []
    if ei.coupling == 'not_measured' or not vt.finite(ei.eta_h):
        reason = vocab.snake_words(common.text(ei.reason)) or 'no period'
        needed = read and _outcome(analysis) not in vt.GATE_VERDICTS
        return _line('2', question, f'Not measured ({reason}).', icon=vc.MISSING if needed else vc.QUIET)
    parts = vt.episode_ei_parts(ei)
    tests = (
        vt.holds(phase_off_deg(ei.phase_deg, 180.0), '<=', ei.phase_tol_deg),
        vt.holds(ei.coherence, '<=', -ei.min_coherence),
        vt.holds(ei.eta_h, '<=', ei.h_residual_max),
    )
    for (name, failed), text, held in zip(EI_NAMES, parts, tests):
        icon = (vc.OK if held else vc.FAIL) if read else vc.QUIET
        children.append(vc.SubCheck(icon, name if held else failed, text))
    if small is not None and vt.finite(small.value):
        held = vt.holds(small.value, str(small.op), small.threshold)
        name = 'forced terms too small to hold ϖ' if held else 'forced terms not too small to hold ϖ'
        children.append(vc.SubCheck(vc.OK if held else vc.QUIET, name, f'ρ_max {_entry_text(small)}'))
    where = f' #{ei.episode_id}' if ei.episode_id is not None else ''
    found = vocab.EI_WORDS.get(ei.coupling, ei.coupling)
    numbers = f'episode{where} at {vt.years(ei.period_yr)}' + (f', {vt.fmt(ei.n_periods)} periods' if vt.finite(ei.n_periods) else '')
    answer = 'Yes.' if ei.coupling == 'zlk' else f'No ({found}).'
    if decides:
        answer = f'No ({found}) where the forced terms are small → {vt.status_head(analysis.verdict.status).lower()}.'
    elif not read:
        answer += ' Not read: the free-ω gate decided.'
    return _line('2', question, answer, numbers, children=children, decides=decides, icon=None if read else vc.QUIET)


def _travel_parts(analysis, row) -> Tuple[str, str, bool]:
    """(where the end travel is read from, its comparison with one turn, whether it is thin) of a row the policy
    decided as too short (`policy.apply_long_period`: σ_f's range from the last turning point of an accepted run, else
    from the window's start, to the window's end; thin within CONFIDENCE_THIN_BAND of one turn, `policy._travel_thin`)."""
    entries = _entries(analysis)
    travel, thin = entries.get(pol.END_TRAVEL_ENTRY), entries.get(pol.END_TRAVEL_THIN_ENTRY)
    geometry = analysis.geometry
    t_last = vt._float(row.get('pol_t_last_tp_yr'))
    start = geometry is not None and vt.finite(t_last) and t_last <= float(geometry.window[0])
    where = 'over the window (no accepted run, so no last turning point)' if start else 'after the last turning point'
    if vt.finite(t_last) and not start:
        where += f' ({vt.years(t_last)})'
    numbers = f'σ_f spans {_turns(travel)} {where}' if travel is not None and vt.finite(travel.value) else ''
    is_thin = thin is not None and thin.result is False
    if is_thin:
        numbers += f'; thin: {_turns(thin)}'
    return where, numbers, is_thin


def _short_line(analysis, row) -> vc.CheckLine:
    """◆ P of a row the policy decided as too short (`pol_decided_by` insufficient_time): MMR's question; a record that
    ends in circulation (σ_f spans at least one turn after the last turning point) is non-resonant, at most medium when
    that travel is thin (W36 F5, 143662 s+g5−g8−s3: the shape's step still says 'not yet')."""
    status = f'!!{vt.status_head(analysis.verdict.status).lower()}!!'
    _, numbers, thin = _travel_parts(analysis, row)
    if str(analysis.verdict.status) == 'non_resonant':
        answer = f'No, but σ_f ends in circulation → {status}'
        if thin:
            answer += ', at most medium: within the thin band of one turn'
    else:
        answer = mmr.POLICY_QUESTIONS[pol.DECIDED_INSUFFICIENT][1].format(status=status)
    return vc.CheckLine(vc.DECIDES, POLICY_NUMBER, POLICY_SHORT_QUESTION, answer, numbers, POLICY_NUMBER, decides=True)


def _policy_line(analysis, row) -> vc.CheckLine:
    """◆ P: a row the policy decided as too short (`_short_line`); else the free-ω gate gave no verdict and the policy
    kept the shape's status (at most medium; at its level when a kinematic libration is excluded, flag
    kinematic_excluded); the gate's numbers that stopped it."""
    if _decided_by(row) == pol.DECIDED_INSUFFICIENT:
        return _short_line(analysis, row)
    words, numbers = _no_verdict_parts(analysis, row)
    if not numbers and _outcome(analysis) == 'short_window':
        cycles = _entries(analysis).get('gate_cycles')
        numbers = f'{GATE_CYCLES_WORDS} {_entry_text(cycles)}' if cycles is not None and vt.finite(cycles.value) else ''
    status = f'!!{vt.status_head(analysis.verdict.status).lower()}!!'
    if vt.has_mark(analysis.verdict, 'kinematic_excluded'):
        answer = f'No ({words}); a kinematic libration is excluded (step 2) → the shape’s status ({status}) is kept, at the level its evidence gives'
    else:
        answer = f'No ({words}) → the shape’s status ({status}) is kept, at most medium'
    return vc.CheckLine(vc.DECIDES, POLICY_NUMBER, POLICY_GATE_QUESTION, answer, numbers, POLICY_NUMBER, decides=True)


def _not_reached(numbers: Sequence[str], words: Sequence[str]) -> vc.CheckLine:
    span = numbers[0] if len(numbers) == 1 else f'{numbers[0]}–{numbers[-1]}'
    return vc.CheckLine(vc.QUIET, span, ' · '.join(words), '— not reached', quiet=True)


def checks(analysis, settings, row, episodes, family: str) -> Tuple[Tuple[vc.CheckLine, ...], str]:
    """(the steps, the deciding one's rule id)."""
    eps = _eps(analysis, episodes)
    step = deciding_step(analysis, row, family)
    q = _q_step(analysis, row, settings, step == 'Q')
    if _unphysical(analysis):
        words = ('shape', STEP_WORDS[f'{family}:1'], STEP_WORDS[f'{family}:2'])
        return (q, _not_reached(('0', '2'), words)), ''
    lines = [q, _shape_step(analysis, row, settings, eps, family, step == '0')]
    if family == 'zlk':
        policy = step == POLICY_NUMBER and _decided_by(row) == pol.DECIDED_GATE_UNAVAILABLE
        later = [_gate_step(analysis, row, settings, step == '1', policy), _ei_step(analysis, row, settings, step == '2')]
    else:
        later = [_secular_centre_step(analysis, row, step == '1'), _phase_step(analysis, row, step == '2')]
    missing = [str(i + 1) for i, line in enumerate(later) if line is None]
    lines += [line for line in later if line is not None]
    if missing:
        words = [STEP_WORDS[f'{family}:{n}'] for n in missing]
        if len(missing) == 2 or missing == ['2']:
            lines.append(_not_reached(missing, words))
        else:  # a step 1 not run between a shown step 0 and step 2
            lines.insert(2, _not_reached(missing, words))
    if step == POLICY_NUMBER:
        lines.append(_policy_line(analysis, row))
    decided = next((line for line in lines if line.decides), None)
    return tuple(lines), decided.rule_id if decided is not None else ''


# --------------------------------------------------------------------------------------------- subtitle and key facts


def _status_words(analysis, row, family: str) -> str:
    v = analysis.verdict
    status, angle, subtype = str(v.status), str(v.angle_status), str(v.angle_subtype)
    if status == 'invalid':
        return 'the orbit is unphysical: no angle is read'
    if status == 'uncertain':
        if subtype == 'phase_unresolved':
            return 'its swing is within the error of the mode phases'
        if 'centre_turned_insufficient' in set(v.rules_fired):
            return 'part of one slow swing of the center: too short to tell'
        if angle == 'drift':
            return f'{NOUN[family]} oscillates about a drifting center'
        return 'too few turning points to tell'
    where = 'over the whole record' if angle == 'libration' else 'in part of the record'
    if family == 'secular':
        if status == 'non_resonant':
            return 'the proper angle circulates'
        return f'the proper angle librates {where}'
    return _zlk_status_words(analysis, where)


def _zlk_status_words(analysis, where: str) -> str:
    v = analysis.verdict
    status, angle, outcome = str(v.status), str(v.angle_status), _outcome(analysis)
    if angle == 'circulation' or status == 'non_resonant' and str(v.reason) == 'circulation':
        return 'ω circulates'
    if status == 'non_resonant':
        if _veto(analysis) == pol.VETO_ZLK_INTEGRAL:
            return 'ω librates, but e and i do not trade as a ZLK cycle'
        return 'ω librates, but only kinematically: the free ω winds'
    if outcome == 'confirmed':
        return f'ω librates {where}; the free ω librates too: confirmed'
    if outcome == 'kinematic':
        return f'ω librates {where}; the free ω may wind (an objection)'
    return f'ω librates {where}; the free-ω gate gives no verdict'


def subtitle(analysis, row, family: str, step: str, planet: str = '') -> vc.Subtitle:
    """The title and its detail line; `planet` names the co-orbital zone ⟨a⟩ lies in (`coorbital_planet`)."""
    name = resonance_words(common.text(row.get('resonance')) or common.text(getattr(analysis.series, 'name', '')))
    title = f'{name} · {_status_words(analysis, row, family)}'
    what = STEP_WORDS.get(step) or STEP_WORDS.get(f'{family}:{step}', '')
    decided = f'decided at step {step} ({what})'
    if step == POLICY_NUMBER and _decided_by(row) == pol.DECIDED_INSUFFICIENT:
        decided = f'decided by the policy: too few turning points, and the record ends in circulation (step {POLICY_NUMBER})'
        if str(analysis.verdict.status) != 'non_resonant':
            decided = f'decided by the policy: the record is too short (step {POLICY_NUMBER})'
    elif step == POLICY_NUMBER:
        decided = f'decided without a gate verdict: the status follows the shape of {NOUN[family]} (step P)'
    if family == 'secular':
        detail = f'secular resonance, proper angle · {decided}'
    else:
        frame = common.text(row.get('zlk_angle_frame')) or 'invariable'
        detail = ' · '.join(p for p in (f'ZLK test on ω ({frame} plane)', f'co-orbital zone of {planet}' if planet else '', decided) if p)
    return vc.Subtitle(title, detail)


def coorbital_planet(row: Mapping[str, Any], settings) -> str:
    """The planet whose co-orbital zone ⟨a⟩ (ctx_a_med_au) lies in, by `settings`' ZLK_COORBITAL_HILL_FACTOR
    (`secular_plots.coorbital_planet_geometric`, the zone panel (d) draws); '' outside every zone."""
    from resonances.plotting.secular_plots import coorbital_planet_geometric

    a = vt._float(row.get('ctx_a_med_au'))
    return coorbital_planet_geometric(a, mmr.panel.config_settings(settings)) if vt.finite(a) else ''


def _frequency(frequencies, name: str):
    return next((item for item in frequencies or () if item.name == name), None)


def _sigma_fact(analysis, row, eps: vt.Episodes, family: str) -> str:
    v = analysis.verdict
    angle = str(v.angle_status)
    span = vt.record_span_yr(analysis)
    decisive = eps.decisive if eps.rows else None
    centre = vt._float((decisive or {}).get('centre_mod_2pi_deg'))
    centre_text = f' · center **{why.centre_text(centre)}**' if vt.finite(centre) else ''
    # the cycles the level counts (input N), the number the Why and the pill print; else the deciding episode's own
    counted = {name: state for name, state, _, _ in common.parse_inputs(row.get('pol_level_inputs'))}.get('N', '')
    cycles = vt._float(counted) if counted else vt._float(row.get('pol_core_cycles'))
    own = ' counted' if counted else ' in the deciding episode'
    cycles_text = f' · {vt.fmt(cycles)} cycle{"" if cycles == 1 else "s"}{own}' if vt.finite(cycles) else ''
    if angle == 'libration':
        # the status reads coverage, not the share: a libration whose first and last turning points lie within the
        # coverage gap of the record's ends covers the record (step 0)
        return f'librates over the whole record ({_span_words(analysis, eps, span)}){cycles_text}{centre_text}'
    if angle == 'episodic':
        n = len(eps.rows)
        return f'librates in {vt.plural(n, "episode")} covering **{vt.percent(eps.covered_fraction())}** of {vt.years(span)}{centre_text}'
    if angle == 'circulation':
        passed = vt._float(row.get('geo_centre_max_pass_rev'))
        return 'circulates' + (f': its center runs {vt.fmt(passed)} turns one way in {vt.years(span)}' if vt.finite(passed) else '')
    if _unphysical(analysis):
        return 'not read (the orbit is unphysical)'
    words = _shape_answer(analysis, row, eps, family).rstrip('.')
    travel = vt._float(row.get('pol_end_travel_rev'))
    if _decided_by(row) == pol.DECIDED_INSUFFICIENT and str(v.status) == 'non_resonant' and vt.finite(travel):
        return f'too few turning points for one cycle; σ_f spans **{vt.fmt(travel)} turns** to the end: it ends in circulation'
    return words[:1].lower() + words[1:]


def _span_words(analysis, eps: vt.Episodes, span: float) -> str:
    """'turning points span 66% of 10 Myr; gaps to both ends ≤ one libration period' (the share of the counted episodes from
    the first to the last turning point; the coverage gaps of the ledger)."""
    words = f'turning points span **{vt.percent(eps.covered_fraction())}** of {vt.years(span)}'
    entries = _entries(analysis)
    start, end = entries.get('coverage_start'), entries.get('coverage_end')
    if start is None or end is None or not all(vt.holds(e.value, str(e.op), e.threshold) for e in (start, end)):
        return words
    period = 'one libration period' if start.threshold == 1 else f'{vt.fmt(start.threshold)} libration periods'
    return f'{words}; gaps to both ends ≤ {period}'


def key_facts(analysis, row, episodes, frequencies, family: str, planet: str = '') -> Tuple[vc.Fact, ...]:
    """The Key facts; `planet` names the co-orbital zone ⟨a⟩ lies in (`coorbital_planet`)."""
    eps = _eps(analysis, episodes)
    noun = NOUN[family]
    facts = [vc.Fact(noun, _sigma_fact(analysis, row, eps, family))]
    decisive = eps.decisive if eps.rows else None
    if decisive is not None and str(analysis.verdict.angle_status) in ('libration', 'episodic', 'insufficient_time'):
        amp = f' · half-amplitude **{_deg(decisive.get("half_amp_med_deg"))}**' if vt.finite(decisive.get('half_amp_med_deg')) else ''
        top = f' (max {_deg(decisive.get("half_amp_max_deg"))})' if vt.finite(decisive.get('half_amp_max_deg')) else ''
        facts.append(vc.Fact('period · amplitude', f'P **{vt.years(decisive.get("period_med_yr"))}**{amp}{top}'))
    if len(eps.rows) > 1:
        items = []
        shown = [r for r in eps.rows if vt._same_id(r.get('episode_id'), analysis.decisive_episode_id)]
        shown = (shown + [r for r in eps.rows if r not in shown])[:EPISODES_SHOWN]
        for episode in sorted(shown, key=lambda r: vt._float(r.get('t_start_yr'))):
            mark = ' ◆' if vt._same_id(episode.get('episode_id'), analysis.decisive_episode_id) else ''
            items.append(
                f'#{vt.fmt(episode.get("episode_id"))} {vt.interval(episode.get("t_start_yr"), episode.get("t_end_yr"))} '
                f'P {vt.years(episode.get("period_med_yr"))} A {_deg(episode.get("half_amp_med_deg"))}{mark}'
            )
        more = len(eps.rows) - len(shown)
        facts.append(vc.Fact('episodes', '; '.join(items) + (f'; {more} more' if more else '')))
    if family == 'zlk':
        facts += _zlk_facts(analysis, row, frequencies)
    else:
        facts += _secular_facts(row, frequencies)
    facts.append(vc.Fact('orbit', _orbit_fact(row, family)))
    facts.append(vc.Fact('overlaps', _overlap_words(row) + (f' · in the co-orbital zone of {planet}' if planet else '')))
    facts.append(vc.Fact('run', _run_fact(row)))
    return tuple(fact for fact in facts if fact.value)


def _overlap_words(row) -> str:
    """`verdict_text.orbit_line` in the body's q–Q words: 'q–Q overlaps that of Jupiter (…)', 'q–Q overlaps no
    planet’s q–Q (…)'; the planets' bands are their own q–Q."""
    line = vt.orbit_line(row).rstrip('.')
    if line.startswith('Radial overlap with '):
        rest = line[len('Radial overlap with ') :]
        names = rest.split(' (')[0]
        those = 'those' if ',' in names else 'that'
        return f'q–Q overlaps {those} of {rest}'
    if line.startswith('Radial overlap: none'):
        return 'q–Q overlaps no planet’s q–Q' + line[len('Radial overlap: none') :]
    return line.replace('Radial overlap: ', 'q–Q overlap: ')


def _zlk_facts(analysis, row, frequencies) -> List[vc.Fact]:
    if _unphysical(analysis):
        return []
    out = []
    free = _frequency(frequencies, 'g − s')
    revolutions, extent = vt._float(row.get('free_omega_revolutions')), vt._float(row.get('free_omega_range_deg'))
    # the period of the mean rate g − s (the FREQUENCIES row), not a libration period of the free ω
    period = f'360°/|g − s| **{vt.years(free.period)}**' if free is not None and vt.finite(free.period) else ''
    gate = _gate(analysis)
    if gate is not None and common.text(gate.detail) == 'coorbital_unfitted_forcing':
        # the planet's own forcing is not in the fitted basis: the split's straight line is no measurement (CLAUDE.md)
        out.append(vc.Fact('free ω (g − s)', 'not a measurement: the free-element model does not apply to a co-orbital'))
        return out
    inside = vt._float(row.get('zlk_gate_free_turns_episodes_max'))
    if vt.finite(extent) and extent >= 360 and vt.finite(revolutions):
        motion = f'**winds**: {vt.fmt(abs(revolutions))} net turns over the record'
        if _outcome(analysis) == 'confirmed' and common.text(gate.detail) == 'episodes' and vt.finite(inside):
            # the gate read inside the episodes (W34): both are true, the record's winding and the episodes' stillness
            motion = (
                f'**winds** over the record ({vt.fmt(abs(revolutions))} net turns) but not inside the episodes (at most '
                f'{vt.fmt(inside)} turns per cycle)'
            )
    elif vt.finite(extent):
        # an unwrapped range below 360° never completes a turn; 'librates' only where the gate confirmed it
        motion = f'full range {_deg(extent)}, never a whole turn'
        if _outcome(analysis) == 'confirmed':
            motion = f'librates: {motion}'
    else:
        motion = 'not measured (no free elements)'
    out.append(vc.Fact('free ω (g − s)', ' · '.join(p for p in (period, motion) if p)))
    zlk = getattr(analysis, 'zlk', None)
    attribution = zlk.attribution if zlk is not None else None
    if attribution is not None and attribution.attributed:
        mode = attribution.mode.split(' ')[0]
        out.append(
            vc.Fact(
                'forcing',
                f'mode {mode}, its forced e |c_{mode}| {vt.fmt(attribution.forcing_e)} · e_free/|c_{mode}| '
                f'{vt.fmt(attribution.free_over_forcing)} (against this term alone)',
            )
        )
    elif _outcome(analysis) == 'kinematic' and attribution is not None:
        out.append(vc.Fact('forcing', f'not attributed ({vocab.snake_words(common.text(attribution.reason))})'))
    return out


def _secular_facts(row, frequencies) -> List[vc.Fact]:
    modes = [item for item in frequencies or () if item.note.startswith('in σ')]
    sigma = _frequency(frequencies, 'σ')
    parts = []
    if modes:
        parts.append('modes in σ (coefficient): ' + ', '.join(f'{item.name} {item.note[len("in σ "):]}' for item in modes))
    if sigma is not None and vt.finite(sigma.rate):
        plus = '+' if sigma.rate > 0 else ''
        parts.append(f'mean rate σ̇ {plus}{_signed(sigma.rate)}″/yr (360°/|σ̇| {vt.years(sigma.period)})')
    return [vc.Fact('resonance', ' · '.join(parts))] if parts else []


def _orbit_fact(row, family: str) -> str:
    parts = []
    a_lo, a_hi = vt._float(row.get('ctx_a_min_au')), vt._float(row.get('ctx_a_max_au'))
    if vt.finite(a_lo) and vt.finite(a_hi):
        parts.append(f'a {_a_range(a_lo, a_hi)} au')
    e_lo, e_hi = vt._float(row.get('ctx_e_min')), vt._float(row.get('ctx_e_max'))
    if vt.finite(e_lo) and vt.finite(e_hi):
        parts.append(f'e {vt.fmt(e_lo)}–{vt.fmt(e_hi)}')
    i_lo, i_hi = vt._float(row.get('ctx_inc_min_deg')), vt._float(row.get('ctx_inc_max_deg'))
    if vt.finite(i_lo) and vt.finite(i_hi):
        plane = ' (invariable plane)' if family == 'zlk' and common.text(row.get('zlk_angle_frame')) in ('', 'invariable') else ''
        parts.append(f'i {vt.fmt(i_lo)}–{vt.fmt(i_hi)}°{plane}')
    return ' · '.join(parts)


def _a_range(lo: float, hi: float) -> str:
    """The range of a with one number of decimals for both ends, as few as tell them apart (from 3 significant digits
    of the larger, at most 6 decimals): '2.788–2.795', '2.79–2.80'; a negative end (an unbound orbit) 'from −506 to
    135'. Formatting only: the numbers are ctx_a_min_au, ctx_a_max_au."""
    if lo < 0 or hi < 0:
        return f'from {_signed(lo)} to {_signed(hi)}'
    size = max(abs(lo), abs(hi))
    decimals = max(0, 2 - math.floor(math.log10(size))) if size > 0 else 0
    while f'{lo:.{decimals}f}' == f'{hi:.{decimals}f}' and decimals < 6:
        decimals += 1
    if f'{lo:.{decimals}f}' == f'{hi:.{decimals}f}':  # no variation to show
        return vt.fmt(lo)
    return f'{lo:.{decimals}f}–{hi:.{decimals}f}'


def _run_fact(row) -> str:
    span, step, cutoff = vt._float(row.get('smp_span_yr')), vt._float(row.get('smp_step_yr')), vt._float(row.get('flt_cutoff_yr'))
    binding = BINDING_WORDS.get(common.text(row.get('flt_cutoff_binding')), vocab.snake_words(common.text(row.get('flt_cutoff_binding'))))
    parts = [
        vt.years(span) if vt.finite(span) else '',
        f'output {vt.years(step)}' if vt.finite(step) else '',
        f'filter cutoff {vt.years(cutoff)}' + (f' (set by {binding})' if binding else '') if vt.finite(cutoff) else '',
    ]
    if common.text(row.get('sec_scale')) == 'centre':
        parts.append('the center scale decided')
    return ' · '.join(p for p in parts if p)


# --------------------------------------------------------------------------------------------- confidence


def _v_words(analysis, row, state: str, family: str) -> str:
    v = analysis.verdict
    if family == 'secular':
        return {'clean': 'error of the mode phases ruled out', 'not_measured': 'error of the mode phases not measured'}.get(state, '')
    outcome = _outcome(analysis)
    if state == 'clean':
        if vt.has_mark(v, 'kinematic_excluded'):
            return 'kinematic libration excluded: forced e small, e-i integral held'
        if outcome == 'confirmed':
            return 'the free ω librates (gate confirmed)'
        return 'no alternative left to measure'
    if state == 'partial':
        return 'free-ω gate confirmed with few cycles to spare'
    if state == 'not_measured':
        words = _no_verdict_parts(analysis, row)[0] if outcome not in ('', 'disabled') else 'the gate did not run'
        return f'free-ω gate: no verdict ({words})'
    return ''


def _long_pill(pill: vc.Pill, analysis, family: str, row) -> Optional[vc.Pill]:
    """A pill in long-period words; None for an input that is no question here (both resonances at once: MMR only).
    The veto's pill stays `verdict_column_common.pill_text`'s ('overruled by …', D9, as the MMR pill)."""
    inputs = {name: state for name, state, _, _ in common.parse_inputs(row.get('pol_level_inputs'))}
    state = inputs.get(pill.key, '')
    text = pill.text
    if pill.key == 'second_capture' and state != 'unresolved':
        return None
    if pill.key == 'M':
        text = {'full': 'amplitude clear of the run break', 'thin': 'amplitude near the run break'}.get(state, text)
    elif pill.key == 'V' and state != 'ambiguous':
        text = _v_words(analysis, row, state, family) or text
    return vc.Pill(pill.icon, text, pill.key, pill.ceiling)


def _missing(analysis, row, family: str) -> List[str]:
    """What an uncertain or invalid row lacks, in words (the chip says the action)."""
    v = analysis.verdict
    angle, subtype = str(v.angle_status), str(v.angle_subtype)
    if str(v.status) == 'invalid':
        return ['a bound orbit: the elements of an unbound one are not read']
    if subtype == 'phase_unresolved':
        return ['a swing beyond the error of the mode phases (measured phases)']
    if 'centre_turned_insufficient' in set(v.rules_fired):
        return ['one full slow swing of the center (it turned back once)']
    if angle == 'drift':
        return ['a center that stays put (it moves one way)']
    if angle == 'insufficient_time':
        return [f'one full cycle of {NOUN[family]} (fewer turning points than a cycle needs)']
    if str(v.reason) == 'gate_unavailable':
        return ['a verdict of the free-ω gate']
    return []


def confidence(analysis, row, family: str) -> vc.Confidence:
    level = str(analysis.verdict.confidence or '')
    if not level:
        return common.confidence(row, '', _missing(analysis, row, family))
    block = common.confidence(row, level)
    found = tuple(pill for pill in (_long_pill(pill, analysis, family, row) for pill in block.pills) if pill is not None)
    return vc.Confidence(level, block.because if level == 'certain' else common.because_text(found), found)


# --------------------------------------------------------------------------------------------- evidence


def _zlk_evidence(analysis, row) -> vc.Evidence:
    rows = []
    zlk = getattr(analysis, 'zlk', None)
    outcome = _outcome(analysis)
    if zlk is not None:
        attribution = zlk.attribution
        if attribution.attributed:
            mode, _, group = attribution.mode.partition(' ')
            cluster = f' cluster {group.strip()}' if group else ''
            rows.append(
                vc.EvidenceRow(
                    vc.OK,
                    'forced vector',
                    f'{mode}{cluster}, |c_{mode}| {vt.fmt(attribution.forcing_e)}; e_free/|c_{mode}| '
                    f'{vt.compare(attribution.free_over_forcing, "<", 1.0)} ' + vc.panels(('j',)),
                )
            )
        elif outcome == 'kinematic':
            rows.append(
                vc.EvidenceRow(vc.MISSING, 'forced vector', f'not attributed ({vocab.snake_words(common.text(attribution.reason))})')
            )
    line = getattr(analysis, 'centre_line', None)
    centre = vt._centre_line(line)
    if centre:
        explained = str(line.origin).startswith(vt.CENTRE_CAUSE_ORIGINS)
        value = re.sub(r'\b([gs]\d+)-([gs]\d+)\b', r'\1 − \2', vocab.snake_words(centre.replace('slow line of the center: ', '')))
        rows.append(vc.EvidenceRow(vc.OK if explained else vc.QUIET, 'center wave', f'{value} {vc.panels(("h",))}'))
    if zlk is not None:
        node = vt._node_lock_words(zlk.node_lock)
        if node:
            node = re.sub(r'ϖ(\d)', lambda m: 'ϖ' + m.group(1).translate(mmr.SUBSCRIPTS), node)  # 'Ω − ϖ5' → 'Ω − ϖ₅'
            rows.append(vc.EvidenceRow(vc.OK if zlk.node_lock.locked else vc.MISSING, 'node lock', node))
        ei = zlk.ei
        if ei is not None:
            # a ZLK coupling (constant √(1−e²) cos i) holds for a circulating ω too: no ✓ for a row that does not librate
            icon = vc.OK if ei.coupling == 'zlk' else (vc.MISSING if ei.coupling == 'not_measured' else vc.QUIET)
            if icon == vc.OK and str(analysis.verdict.status) == 'non_resonant':
                icon = vc.QUIET
            rows.append(vc.EvidenceRow(icon, 'e-i coupling', vt._ei_line(ei).replace('e-i ', '', 1)))
    circulates = str(analysis.verdict.angle_status) == 'circulation'
    return vc.Evidence('Evidence: what moves with ω?' if circulates else 'Evidence: what holds ω?', tuple(rows))


def _secular_evidence(analysis, row, frequencies) -> vc.Evidence:
    rows = []
    span = vt.record_span_yr(analysis)
    if vt.finite(span) and span > 0 and frequencies:
        resolution = ARCSEC_PER_TURN / span
        for label, period, low, high in vt._seen_periods(analysis, row, False):
            match, nearest = vt._identify(period, low, high, resolution, frequencies)
            if match:
                names = ' or '.join(_period_of(frequencies, name) for name in match.split(' or '))
                rows.append(
                    vc.EvidenceRow(vc.OK, 'period seen', f'{vt.years(period)} ({label}) matches the period of {names} {vc.panels(("a",))}')
                )
            else:
                near = f'; the closest in frequency is {_period_of(frequencies, nearest)}' if nearest else ''
                rows.append(
                    vc.EvidenceRow(
                        vc.MISSING,
                        'period seen',
                        f'{vt.years(period)} ({label}): none of the listed frequencies within its range ± 1/T{near}',
                    )
                )
    # the mean rate of σ is a Key fact (resonance); here only where a period seen is matched against it
    centre = vt._centre_line(getattr(analysis, 'centre_line', None))
    if centre:
        rows.append(
            vc.EvidenceRow(
                vc.QUIET, 'center wave', vocab.snake_words(centre.replace('slow line of the center: ', '')) + ' ' + vc.panels(('h',))
            )
        )
    return vc.Evidence('Evidence: what makes the cycles?', tuple(rows))


def _period_of(frequencies, name: str) -> str:
    """'σ from mean rates, 594 kyr', 'g5 Jup, 304 kyr': a FREQUENCIES row by its name and period."""
    item = _frequency(frequencies, name)
    if item is None or not vt.finite(item.period):
        return name
    source = ' from mean rates' if item.note == 'mean rates' else ''
    return f'{name}{source}, {vt.years(item.period)}'


def evidence(analysis, row, frequencies, family: str) -> Tuple[vc.Evidence, ...]:
    if _unphysical(analysis):
        return ()
    block = _zlk_evidence(analysis, row) if family == 'zlk' else _secular_evidence(analysis, row, frequencies)
    return (block,) if block.rows else ()


# --------------------------------------------------------------------------------------------- caveats

# long-period flags whose sentence lives here (verdict_vocab.FLAG_CAVEATS covers the shared ones); a flag the level
# reads (gate_unavailable, kinematic_excluded, ei_integral_not_held, veto_thin) is a pill, not a caveat
LONG_CAVEATS = {
    'three_tp_episode': (('resonant', 'transient'), 'The deciding libration rests on three turning points: one cycle.'),
    'veto_needs_length': (
        ('resonant', 'transient', 'non_resonant'),
        'The free-ω gate needs a longer record to give its verdict.',
    ),
    # the registry's sentences carry the edge's distance in periods; a long-period edge is read about the mean centre
    # (not the episode's gap that step 0 prints), so the long rows say it without the number
    'entry_suspected': (
        ('resonant', 'transient'),
        'An entry is suspected: before the first librating episode{first}, {noun} last moves more than 180° from that '
        'episode’s mean center too near the record’s start to show what came before, so the libration may have begun '
        'before the record.',
    ),
    'exit_suspected': (
        ('resonant', 'transient'),
        'An exit is suspected near the record’s end, so the libration may continue past the record.',
    ),
}


def _first_episode(analysis, episodes) -> str:
    """' (from 66.4 kyr)': the start of the first librating episode (its first turning point), '' without one."""
    starts = [vt._float(r.get('t_start_yr')) for r in _eps(analysis, episodes or ()).rows]
    starts = [t for t in starts if vt.finite(t)]
    return f' (from {vt.years(min(starts))})' if starts else ''


def caveats(analysis, row, family: str, episodes: Sequence[Mapping[str, Any]] = ()) -> Tuple[str, ...]:
    v = analysis.verdict
    status = str(v.status)
    out: List[str] = []
    flags = [str(flag) for flag in (v.flags or ())]
    filled = mmr.slots(row)
    for flag in flags:
        if flag in LONG_CAVEATS:  # the long-period sentence overrides the registry's (`verdict_vocab.FLAG_CAVEATS`)
            applies, template = LONG_CAVEATS[flag]
            text = template.format(noun=NOUN[family], first=_first_episode(analysis, episodes)) if status in applies else ''
        else:
            text = mmr.flag_caveat(flag, status, filled)
        if text and text not in out:
            out.append(text)
    estimates = row.get('ep_centre_estimates')
    if str(v.angle_status) in ('libration', 'episodic') and vt.finite(estimates) and int(float(estimates)) == 1:
        out.append('Only one independent center estimate: one cycle cannot show a fixed center.')
    if family == 'zlk':
        out += _zlk_caveats(analysis, row)
    return tuple(out)


def _zlk_caveats(analysis, row) -> List[str]:
    v = analysis.verdict
    out = []
    coupling = common.text(row.get('zlk_ei_coupling'))
    if (
        str(v.status) in ('resonant', 'transient')
        and coupling not in ('', 'zlk', 'not_measured')
        and not vt.has_mark(v, 'ei_integral_not_held')
        and _outcome(analysis) == 'confirmed'
    ):
        found = vocab.EI_WORDS.get(coupling, vocab.snake_words(coupling))
        out.append(f'Over the whole record e and i do not trade as a ZLK cycle at the deciding period ({found}); the free-ω gate decided.')
    gate = _gate(analysis)
    ei = getattr(gate, 'episode_ei', None) if gate is not None else None
    if ei is not None and ei.coupling not in ('zlk', 'not_measured') and vt.finite(ei.eta_h) and not _ei_read(analysis, row):
        failed = f'In the deciding episode e and i do not trade as a ZLK cycle (η_H {vt.compare(ei.eta_h, "<=", ei.h_residual_max)}).'
        if _veto(analysis) in (pol.VETO_ZLK_KINEMATIC, pol.VETO_ZLK_FREE_E):  # a veto of the free ω made the row non-resonant
            out.append(f'{failed} This agrees with the verdict, but the level does not read it: the winding free ω already decided.')
        else:
            out.append(f'{failed} The level does not read it: the free-ω gate did not run on this row.')
    zlk = getattr(analysis, 'zlk', None)
    attribution = zlk.attribution if zlk is not None else None
    if attribution is not None and attribution.attributed and '(' in attribution.mode:
        mode, _, group = attribution.mode.partition(' ')
        out.append(
            f'The forcing {mode} is fitted as the cluster {group.strip("()")}: the free/forced split, and so the gate, rest on that fit.'
        )
    return out


# --------------------------------------------------------------------------------------------- numbers


def _gate_comparison(analysis, row, step: Optional[vc.CheckLine]) -> str:
    """The DECISIVE line of a free-ω gate that set the status (◆ on step 1), as a comparison from the gate's columns and
    the ledger: 'free-ω gate: kinematic · ρ_max 0.431 · cycles of ω 83.3 ≥ 3 [SZ.G]' (a confirmed gate: the free ω's
    range instead of ρ_max; a veto: its own comparisons, the ◆ step's numbers). '' for any other ◆."""
    outcome = _outcome(analysis)
    if step is None or step.number != '1' or outcome not in vt.GATE_VERDICTS:
        return ''
    parts = [f'free-ω gate: {outcome}']
    rho, free = vt._float(row.get('zlk_gate_rho_max')), vt._float(row.get('free_omega_range_deg'))
    if outcome == 'kinematic' and vt.finite(rho):
        parts.append(f'ρ_max {vt.fmt(rho)}')
    elif outcome == 'confirmed' and common.text(_gate(analysis).detail) == 'episodes':
        turns = vt._float(row.get('zlk_gate_free_turns_episodes_max'))
        parts.append('inside the episodes' + (f': free ω turns per cycle at most {vt.fmt(turns)}' if vt.finite(turns) else ''))
    elif outcome == 'confirmed' and vt.finite(free) and free < 360:
        parts.append(f'range of the free ω {_deg(free)}')
    cycles = _entries(analysis).get('gate_cycles')
    numbers = vc.plain(step.numbers) if step.numbers else ''
    if numbers:
        parts.append(numbers)
    elif cycles is not None and vt.finite(cycles.value):
        parts.append(f'{GATE_CYCLES_WORDS} {_entry_text(cycles)}')
    return ' · '.join(parts) + f' [{step.rule_id or "SZ.G"}]'


def _decisive_lines(analysis, row, lines: Sequence[vc.CheckLine], settings) -> Tuple[str, ...]:
    """DECISIVE: the argument that set the status with its ids: a gate that decided as one comparison
    (`_gate_comparison`), any other ◆ as `long_period_decisive_line` and the ◆ step's comparison; then the kinematic
    exclusion when it held the level."""
    if _unphysical(analysis):
        return (vt.UNPHYSICAL_DECISIVE,)
    step = next((line for line in lines if line.decides), None)
    gate = _gate_comparison(analysis, row, step)
    if gate:
        out = [gate]
    elif step is not None and step.number == POLICY_NUMBER and _decided_by(row) == pol.DECIDED_INSUFFICIENT:
        out = [SHORT_DECISIVE.get(str(analysis.verdict.status), SHORT_DECISIVE[''])]
    else:
        text, ids, _ = vt.long_period_decisive_line(analysis, row)
        out = [text + (f' [{", ".join(ids)}]' if ids else '')]
    if step is not None and step.numbers and not gate:
        out.append(f'step {step.number}: {vc.plain(step.numbers)}' + (f' [{step.rule_id}]' if step.rule_id else ''))
    excluded = vt.kinematic_excluded_line(analysis, settings)
    if excluded:
        out.append(excluded + ' [SZ.G]')
    return tuple(out)


def _cell(column: str, value: Any) -> str:
    from resonances.plotting.combined_numbers import format_cell, unit_space

    if column.endswith('_yr') and vt.finite(value):
        return vt.years(value)
    if column.endswith('_deg') and vt.finite(value):
        return f'{format_cell(column, value)}°'
    if column == 'zlk_gate':
        return vocab.snake_words(common.text(value))
    if column in ('zlk_ei_coupling', 'zlk_ei_episode_coupling'):
        return vocab.EI_WORDS.get(common.text(value), vocab.snake_words(common.text(value)))
    text = format_cell(column, value)
    return text if text == '—' else text + unit_space(column)


def _rows(row, items: Sequence[Tuple[str, str]]) -> Tuple[Tuple[str, str], ...]:
    from resonances.plotting.combined_numbers import missing

    return tuple((label, _cell(column, row.get(column))) for label, column in items if not missing(row.get(column)))


def _range(row, label: str, lo: str, hi: str, unit: str = '', med: str = '') -> Optional[Tuple[str, str]]:
    from resonances.plotting.combined_numbers import format_cell, missing

    if missing(row.get(lo)) or missing(row.get(hi)):
        return None
    low, high = format_cell(lo, row.get(lo)), format_cell(hi, row.get(hi))
    if med and not missing(row.get(med)):
        return label, f'{low} / {format_cell(med, row.get(med))} / {high}{unit}'
    return label, f'{low} – {high}{unit}'


# the free rows of FREQUENCIES (`long_period_frequencies`) whose meaning needs the planet's forcing in the fitted basis
FREE_ROWS = ('g', 's', 'g − s')
UNFITTED_NOTE = 'not free: {planet}’s forcing not fitted'


def _unfitted(analysis) -> bool:
    """A co-orbital of a planet whose own modes are not in the fitted basis (CLAUDE.md: the free-element method does not
    apply; the gate's outcome unavailable, detail coorbital_unfitted_forcing)."""
    gate = _gate(analysis)
    return gate is not None and common.text(gate.detail) == 'coorbital_unfitted_forcing'


def _frequency_group(frequencies, family: str, unfitted_planet: str = '') -> Optional[vc.NumberGroup]:
    """The FREQUENCIES table; `unfitted_planet` (an unfitted co-orbital) notes on g, s and g − s that they are no free
    rates: the planet's own forced vector is not in the fit, so g − s is no measured free-ω rate."""
    from resonances.plotting.secular_plots import FREQUENCY_GROUP, _rate_text

    if not frequencies:
        return None
    note = '|c|, note' if family == 'zlk' else 'note'
    # a ZLK row's |c| is the fitted forced amplitude: of e for a g mode, of sin(i/2) for an s mode (`free_elements`)
    title = FREQUENCY_GROUP[:-1] + '; |c|: forced e of a g row, forced sin(i/2) of an s row)' if family == 'zlk' else FREQUENCY_GROUP

    def _note(item) -> str:
        if unfitted_planet and item.name in FREE_ROWS:
            return UNFITTED_NOTE.format(planet=unfitted_planet)
        return item.note

    cells = tuple(
        (
            item.name,
            item.rate_text if item.rate_text is not None else _rate_text(item.rate, item.constant),
            vt.years(item.period),
            _note(item),
        )
        for item in frequencies
    )
    return vc.NumberGroup(title, table=('', '″/yr', 'period', note), cells=cells)


def _turn_rows(analysis, row) -> Tuple[Tuple[str, str], ...]:
    """The two counts of the free ω's turns, each under its own label: the net turns of the free elements over the record
    (free_omega_revolutions, the one Key facts and Why print) and the turns at its mean rate |g − s|·T
    (`verdict_column_why.mean_rate_turns`); none for an unfitted co-orbital (no measurement there)."""
    if _unfitted(analysis):
        return ()
    out = []
    net = vt._float(row.get('free_omega_revolutions'))
    if vt.finite(net):
        out.append(('net turns of the free ω (record)', vt.fmt(abs(net))))
    mean = why.mean_rate_turns(analysis, row)
    if mean:
        out.append(('turns at the mean rate |g − s|·T', mean))
    return tuple(out)


def number_groups(analysis, row, frequencies, family: str) -> Tuple[vc.NumberGroup, ...]:
    from resonances.plotting.combined_numbers import missing

    groups = []
    planet = (vt._coorbital_planet(row) or 'the planet') if family == 'zlk' and _unfitted(analysis) else ''
    frequency = _frequency_group(frequencies, family, planet) if not _unphysical(analysis) else None
    noun = NOUN[family]
    libration = _rows(
        row,
        (
            ('center', 'ep_centre_mod_2pi_deg'),
            ('period P', 'ep_period_med_yr'),
            ('half-amplitude median', 'ep_half_amp_med_deg'),
            ('half-amplitude max', 'ep_half_amp_max_deg'),
            ('turning points', 'ep_n_tp'),
            ('independent center estimates', 'ep_centre_estimates'),
            ('gap to the start', 'ep_coverage_start_gap_P'),
            ('gap to the end', 'ep_coverage_end_gap_P'),
            ('turning points of the record', 'geo_n_tp'),
            ('center’s largest one-way pass', 'geo_centre_max_pass_rev'),
        ),
    )
    if libration:
        episode = any(column.startswith('ep_') and not missing(row.get(column)) for column in row)
        groups.append(vc.NumberGroup(f'LIBRATION OF {noun} (deciding episode)' if episode else f'SHAPE OF {noun}', libration))
    if family == 'zlk' and not _unphysical(analysis):
        gate = _rows(
            row,
            (
                ('outcome', 'zlk_gate'),
                ('cycles of ω T / P_cyc', 'zlk_gate_n_cycles'),
                ('cycles needed', 'zlk_gate_min_cycles'),
                ('period read', 'zlk_gate_period_yr'),
                ('free e / forced e, max ρ_max', 'zlk_gate_rho_max'),
                ('e-i over the record', 'zlk_ei_coupling'),
                ('e-i in the deciding episode', 'zlk_ei_episode_coupling'),
            ),
        )
        gate += _turn_rows(analysis, row)
        if gate:
            groups.append(vc.NumberGroup('FREE-ω GATE', gate))
    else:  # D7: the secular e-response is a column only (Q11): here and in the footer, never a caveat
        response = _rows(row, (('elements', 'rsp_variables'), ('R² of e', 'rsp_e_R2'), ('R² of sin(i/2)', 'rsp_i_R2')))
        if response:
            groups.append(vc.NumberGroup('RESPONSE OF e AND i (recorded only; no status reads it)', response))
    orbit = [
        _range(row, 'a', 'ctx_a_min_au', 'ctx_a_max_au', ' au', med='ctx_a_med_au'),
        _range(row, 'e', 'ctx_e_min', 'ctx_e_max'),
        _range(row, 'i (invariable)' if family == 'zlk' else 'i', 'ctx_inc_min_deg', 'ctx_inc_max_deg', '°'),
    ]
    orbit = tuple(item for item in orbit if item is not None)
    if orbit:
        groups.append(vc.NumberGroup('ORBIT (min / med / max)', orbit))
    if frequency is not None:  # where the MMR column has REFERENCE PERIODS: one order of groups for every family
        groups.append(frequency)
    run = list(
        _rows(
            row,
            (('span', 'smp_span_yr'), ('output step', 'smp_step_yr'), ('cutoff P_c', 'flt_cutoff_yr'), ('scale', 'sec_scale')),
        )
    )
    binding = common.text(row.get('flt_cutoff_binding'))
    if binding:
        run.append(('cutoff set by', BINDING_SHORT.get(binding, vocab.snake_words(binding))))
    if run:
        groups.append(vc.NumberGroup('STEP, FILTER', tuple(run)))
    return tuple(groups)


# --------------------------------------------------------------------------------------------- footer and column


def footer(analysis, row, decided_rule: str, family: str) -> Tuple[str, ...]:
    items = list(common.footer(analysis.verdict, row, DOCS))
    extra = []
    if decided_rule and not common.text(row.get('rec_rule')):
        extra.append(f'rule {decided_rule}')
    if family == 'zlk' and _outcome(analysis):
        extra.append(
            f'gate {_outcome(analysis)}' + (f' ({common.text(_gate(analysis).detail)})' if common.text(_gate(analysis).detail) else '')
        )
    response = common.text(analysis.verdict.checks.get('check_secular_e_response', '')) if family == 'secular' else ''
    if response:
        extra.append(f'e-response {response} (recorded only, Q11)')
    return tuple(items[:1] + extra + items[1:])


def long_column(analysis, settings, row: Mapping[str, Any], episodes: Sequence[Mapping[str, Any]], frequencies, family: str) -> vc.Column:
    """The right column of a secular ('secular') or Lidov-Kozai ('zlk') row: `analysis` a `LongPeriodAnalysis`, `row` its
    summary row with the figure's extra columns, `episodes` its episodes.csv rows, `frequencies` the FREQUENCIES rows
    (`secular_plots.long_period_frequencies`); the Why is `verdict_column_why.long_why`."""
    row = dict(row)
    family = 'zlk' if family in ('zlk', 'lidov_kozai') else 'secular'
    lines, rule = checks(analysis, settings, row, episodes, family)
    lines = common.tidy_checks(lines)
    step = next((line.number for line in lines if line.decides), '0')
    block = confidence(analysis, row, family)
    planet = coorbital_planet(row, settings)
    return vc.Column(
        family=family,
        header=common.header(analysis.verdict, row, family, block),
        subtitle=subtitle(analysis, row, family, step, planet),
        key_facts=key_facts(analysis, row, episodes, frequencies, family, planet),
        why=why.long_why(analysis, settings, row, episodes),
        confidence=block,
        checks=lines,
        evidence=evidence(analysis, row, frequencies, family),
        caveats=caveats(analysis, row, family, episodes),
        numbers=vc.Numbers(_decisive_lines(analysis, row, lines, settings), number_groups(analysis, row, frequencies, family)),
        footer=footer(analysis, row, rule, family),
        decided_rule=rule,
    )


def figure_column(analysis, row=None, free_elements=None, settings=None) -> vc.Column:
    """The column the secular/ZLK verdict figure draws, exactly as `secular_plots.draw_long_period_combined` builds it:
    `row` adds columns to the analysis' summary row (as the figure's `row`), `free_elements` (ZLK) the body's
    FreeElements; `settings` default: from the configuration. `verdict_column.column_text` writes it."""
    from resonances.plotting.mmr_verdict_panel import config_settings
    from resonances.plotting.secular_plots import long_period_frequencies

    settings = config_settings(settings)
    summary = {**analysis.summary_row(), **(row or {})}
    episodes = analysis.episode_rows()
    frequencies = long_period_frequencies(analysis, summary, free_elements)
    family = 'zlk' if str(analysis.verdict.resonance_type) == 'lidov_kozai' else 'secular'
    return long_column(analysis, settings, summary, episodes, frequencies, family)
