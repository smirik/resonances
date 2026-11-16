"""The blocks of the right column that every family fills the same way (`verdict_column`): the header with its action
chip, the Confidence block from the policy's level inputs (`policy.level_inputs`, column `pol_level_inputs`) and the
footer of raw codes. The MMR and secular/ZLK builders add the family's own blocks."""

from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from resonances.plotting import verdict_column as vc
from resonances.plotting.verdict_text import finite, years
from resonances.plotting.verdict_vocab import GLANCE_OVERRIDES, GLANCE_PANELS, MMR_MARGIN_NAMES
from resonances.resonance.analysis.verdict import CONFIDENCE_ORDER

ACTION_WORDS = {
    'none': (vc.CHIP_NONE, 'no action needed'),
    'review': (vc.CHIP_REVIEW, 'review figure + numbers'),
}

# what each input of the level is called on its pill (`pill_text`); the policy's names: policy.level_inputs
OBJECTION_WORDS = {
    'force_law': 'the force law fails',
    'potential': 'the energy does not fit the potential',
    'identity': 'σ̇ does not follow a (identity fails)',
    'strength': 'the strength fails',
    'onset': 'an earlier a-wave',
    'strength_scale': 'the strength scale is far too small',
    'force_line': 'the force line fails',
    'owner': 'another resonance may own the libration',
    'kinematic': 'a kinematic bound is possible',
    'zlk_gate_kinematic': 'the free ω may wind',
    'zlk_free_e': 'the free e is below the forced e',
    'zlk_ei_integral': 'e and i do not trade at constant √(1−e²) cos i',
}
FORCE_WORDS = {'L4': 'force law', 'R0': 'potential', 'F': 'strength'}
# a veto in words, for its pill and the Why's limit phrase (the code stays in the footer); an owner is a librating
# resonance that makes the episodes, never called a driver (a driver circulates: docs/physics/response.md#roles)
VETO_WORDS = {
    'owner': 'another resonance that owns the libration',
    'kinematic': 'the kinematic bound',
    'forced_wave': 'the background’s forced wave',
    'strength': 'the strength',
    'zlk_kinematic': 'the winding free ω',
    'zlk_free_e': 'the free e below the forced e',
    'zlk_integral': 'the broken e–i integral',
}


def _number(text: str) -> str:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return text
    return f'{value:.3g}' if math.isfinite(value) else 'none'


def parse_inputs(text: Any) -> List[Tuple[str, str, str, str]]:
    """The column `pol_level_inputs` ('N:4.12>=3:certain;M:thin:high') as (name, state, detail, ceiling)."""
    out = []
    for item in str(text or '').split(';'):
        if item.count(':') < 2:
            continue
        name, middle, ceiling = item[: item.index(':')], item[item.index(':') + 1 : item.rindex(':')], item[item.rindex(':') + 1 :]
        cut = next((i for i, ch in enumerate(middle) if ch in '<>=('), len(middle))
        out.append((name, middle[:cut], middle[cut:], ceiling))
    return out


# the pill words of the inputs whose state alone says it (state -> words; '' = any other state)
STATE_WORDS = {
    'M': {'full': 'clear of the separatrix', 'thin': 'close to the separatrix', '': 'separatrix margin not measured'},
    'S': {'full': 'filter-stable', '': 'filter-sensitive'},
    'V': {
        'clean': 'alternatives ruled out',
        'partial': 'alternatives partly ruled out',
        'not_measured': 'a needed alternative not measured',
    },
    'second_capture': {'unresolved': 'both resonances at once: not ruled out', '': 'both resonances at once: ruled out'},
    'candidate': {'drawn': 'a candidate episode (drawn faint)', '': 'no candidate episode'},
    'status': {'': 'near the separatrix: at most medium'},
    'thin_veto': {'': 'a thin veto: at most medium'},
    'end_travel': {'': 'ends in circulation by a thin margin of one turn: at most medium'},
}


def _cycles(state: str) -> str:
    return 'cycle' if _number(state) == '1' else 'cycles'


# how N was counted (the basis in the column confidence_basis, `policy.Evidence.basis`), its grey gloss: N is
# (turning points − 1)/2 plus the covered edges in periods (`policy.episode_cycles`), core the same without the edges
N_BASIS_WORDS = {
    'episode': 'from the deciding episode’s turning points, covered edges added',
    'cover': 'from the deciding episode’s turning points, covered edges added',
    'union': 'summed over the counted episodes',
    'leg': 'of the 1:1 legs',
    'member': 'the librating member’s',
}


def _bar(detail: str, ceiling: str) -> str:
    """A level bar in words: '>=3' with ceiling certain -> '≥ 3 allows certain'; '<2' with ceiling medium -> 'high
    needs ≥ 2' (the bar opens the level above the ceiling)."""
    if detail.startswith('>='):
        return f'≥ {detail[2:]} allows {ceiling}' if ceiling else f'≥ {detail[2:]}'
    if detail.startswith('<') and ceiling in CONFIDENCE_ORDER and CONFIDENCE_ORDER.index(ceiling) > 0:
        return f'{CONFIDENCE_ORDER[CONFIDENCE_ORDER.index(ceiling) - 1]} needs ≥ {detail[1:]}'
    return detail.replace('>=', '≥ ').replace('<', '< ')


def _n_basis(row: Mapping[str, Any]) -> str:
    basis = re.search(r'\bN=[^;(]*\(([^)]*)\)', str(row.get('confidence_basis') or ''))
    return N_BASIS_WORDS.get(basis.group(1), '') if basis else ''


def _n_words(state: str, detail: str, ceiling: str, row: Mapping[str, Any]) -> str:
    """N in words; a state that is no number ('nan': no episode to count, `policy._librating_inputs`) is 'no cycles',
    which the level reads as 0."""
    if _number(state) == 'none':
        counted, basis = 'no cycles counted', 'no episode'
    else:
        counted, basis = f'{_number(state)} {_cycles(state)} counted', _n_basis(row)
    gloss = '; '.join(part for part in (basis, _bar(detail, ceiling) if detail else '') if part)
    return f'{counted} [[{gloss}]]' if gloss else counted


def _core_words(state: str, detail: str, ceiling: str) -> str:
    if _number(state) == 'none':  # no deciding episode: the level reads the core as 0 cycles
        head, note = 'no cycles between turning points', 'no deciding episode'
    else:
        head, note = f'{_number(state)} {_cycles(state)} between the deciding episode’s turning points', 'edges not added'
    gloss = '; '.join(part for part in (note, _bar(detail, ceiling) if detail else '') if part)
    return f'{head} [[{gloss}]]'


def _circulation_words(state: str, row: Mapping[str, Any]) -> str:
    """The circulation input in what the policy reads (`policy._mmr_circulation_clear`, `_long_circulation_clear`):
    clear = no turning point at any cutoff (×½, ×1, ×2) and the center passes whole turns (C1; P of a secular or ZLK
    row; C5 of an MMR alias, where C1 is not measured); not clear names the part that is missing, from the columns
    geo_n_tp_cut_half, geo_n_tp, geo_n_tp_cut_double, then mrg_C5, mrg_C1 or mrg_P."""
    if state == 'clear':
        return 'clear circulation [[no turning point at any cutoff]]'
    columns = (('½', 'geo_n_tp_cut_half'), ('1', 'geo_n_tp'), ('2', 'geo_n_tp_cut_double'))
    counts = [(scale, row.get(column)) for scale, column in columns]
    seen = [f'{_number(value)} at cutoff ×{scale}' for scale, value in counts if finite(value) and float(value) > 0]
    if seen:
        return f'σ turns back at some filter cutoff [[turning points: {", ".join(seen)}]]'
    if any(not finite(value) for _, value in counts):
        return 'turning points not measured at every cutoff'
    if text(row.get('angle_subtype')) == 'alias':
        alias = text(row.get('mrg_C5'))
        return f'{MMR_MARGIN_NAMES["C5"]}: ' + ('not measured' if alias in ('', 'not_measured') else 'not established')
    long_period = text(row.get('resonance_type')) in ('secular', 'lidov_kozai')
    if text(row.get('mrg_P' if long_period else 'mrg_C1')) in ('', 'not_measured'):
        return 'center turns not measured'
    return 'its center passes too few whole turns for a clear circulation'


def _objection_words(row: Mapping[str, Any]) -> str:
    checks = [c for c in str(row.get('pol_objection_check', '') or '').split(';') if c]
    return 'objection: ' + ('; '.join(OBJECTION_WORDS.get(c, c) for c in checks) or 'a measured objection')


def _force_words(state: str, row: Mapping[str, Any]) -> str:
    if state != 'shown':
        return 'own force not shown'
    shown = [FORCE_WORDS[i] for i in force_shown(row)]
    return f'own force shown ({", ".join(shown)})' if shown else 'own force shown'


THIN_VETO = ' (by a thin margin)'  # was '(thin)': the veto's comparison passes inside the thin band


def pill_text(name: str, state: str, detail: str, row: Mapping[str, Any], ceiling: str = '') -> str:
    """The words of one input's pill (no icon). `row` = the summary row (for the objection's check, the force, how N
    was counted and the turning points of a circulation); `ceiling` the level the input allows (its bar's words)."""
    if name == 'N':
        return _n_words(state, detail, ceiling, row)
    if name == 'core':
        return _core_words(state, detail, ceiling)
    if name == 'circulation':
        return _circulation_words(state, row)
    if name == 'V' and state == 'ambiguous':
        return _objection_words(row)
    if name == 'F':
        return _force_words(state, row)
    if name == 'veto':
        veto = detail.strip('()')
        return f'overruled by {VETO_WORDS.get(veto, veto)}' + (THIN_VETO if state == 'thin' else '')
    if name == 'table':
        return f'level fixed at {state} by the rule that decided'
    words = STATE_WORDS.get(name)
    if words is None:
        return f'{name}: {state}'
    return words.get(state, words.get('', state))


def force_shown(row: Mapping[str, Any]) -> Tuple[str, ...]:
    """The force checks that pass (L4, R0, F) from the summary row's check columns."""
    checks = {'check_mmr_a_response': 'L4', 'check_mmr_potential': 'R0', 'check_mmr_strength': 'F'}
    return tuple(rule_id for column, rule_id in checks.items() if str(row.get(column, '')) == 'pass')


def _rank(level: str) -> int:
    return CONFIDENCE_ORDER.index(level) if level in CONFIDENCE_ORDER else len(CONFIDENCE_ORDER)


def best_state(name: str, state: str, ceiling: str) -> bool:
    """Whether an input is in its best state: it allows certain (N and core above the certain bar, M and S full, V clean,
    F shown, candidate none, circulation clear, both resonances at once ruled out), or it is a full veto (a veto never
    allows more than high)."""
    return ceiling == 'certain' or (name == 'veto' and state == 'full')


def pills(row: Mapping[str, Any], level: str) -> Tuple[vc.Pill, ...]:
    """One pill per input of the level: ✗ an objection the level reads (V ambiguous), ? an input not measured that holds
    the level, ⚠ any other input that holds it (ceiling = the level, below certain), ✓ an input in its best state
    (`best_state`), – a limited input that does not hold the level (another input holds it lower; its words say where it
    alone would cap the level); worst first."""
    out = []
    inputs = parse_inputs(row.get('pol_level_inputs'))
    veto = next((detail.strip('()') for name, _, detail, _ in inputs if name == 'veto'), '')
    for name, state, detail, ceiling in inputs:
        if name == 'second_capture' and state == 'none' and veto != 'owner':
            continue  # 'both resonances at once' is a test of an owner: without one it was never run
        holds = level != 'certain' and _rank(ceiling) >= _rank(level)  # its ceiling is the level (or below it)
        words = pill_text(name, state, detail, row, ceiling)
        if name == 'V' and state == 'ambiguous':
            icon = vc.FAIL
        elif holds and state == 'not_measured':
            icon = vc.MISSING
        elif holds:
            icon = vc.THIN
        elif best_state(name, state, ceiling):
            icon = vc.OK
        else:
            icon = vc.QUIET
            if _rank(ceiling) < len(CONFIDENCE_ORDER) and f'at most {ceiling}' not in words:
                words = f'{words} — alone it would cap the level at {ceiling}'
        out.append(vc.Pill(icon, words, name, ceiling))
    order = (vc.FAIL, vc.MISSING, vc.THIN, vc.QUIET, vc.OK)
    return tuple(sorted(out, key=lambda pill: order.index(pill.icon)))


def _formula(inner: str) -> bool:
    """Whether a parenthesis holds a formula ('a − a*'), not a gloss: an operator, no digit and no word of 4 letters."""
    has_word = any(len(word) >= 4 and word.isalpha() for word in inner.split())
    return any(op in inner for op in '−+*=') and not any(ch.isdigit() for ch in inner) and not has_word


def unnest(text: str, outer: bool = False) -> str:
    """`text` without a gloss in parentheses inside parentheses: 'x (a (b) c)' -> 'x (a, b c)'; `outer`: the text goes
    into gray parentheses of its own (`[[…]]`), so its top-level glosses are flattened too. A formula's parentheses
    stay: 'sin(σ − φ₀)', '√(1−e²)', '(T₀ theory/T₀)²', '(1/8)φ', 'c (a − a*)', and so do panel letters '(j)'."""
    pairs, stack = {}, []
    for i, ch in enumerate(text):
        if ch == '(':
            stack.append(i)
        elif ch == ')' and stack:
            pairs[stack.pop()] = i
    if stack:
        return text
    out = list(text)
    depth = {i: sum(1 for j, k in pairs.items() if j < i < k) + (1 if outer else 0) for i in pairs}
    for i, j in pairs.items():
        after, before = text[j + 1 : j + 2], text[:i].rstrip()
        if depth[i] < 1 or i == 0 or text[i - 1] != ' ' or before.endswith('='):
            continue
        if after.isalnum() or (after and after in '²/') or _formula(text[i + 1 : j]) or _PANELS.match(text[i + 1 : j]):
            continue
        if before[-1:] in ',;:':
            out[i] = ''
        else:
            out[i - 1], out[i] = ',', ' '
        out[j] = ''
    return ''.join(out)


WORST_FIRST = (vc.FAIL, vc.MISSING, vc.THIN, vc.OK)


def worst(children: Sequence[vc.SubCheck], read: Sequence[str] = ()) -> str:
    """A step's icon: the worst of its sub-checks (✓ when none is worse); with `read` (the ids the step's answer
    reads), a ? only from those: a gap the answer did not need is no gap of the step (2017 UM52: O1 cannot be measured
    when the libration starts the record, while the own-force answer stands on L4 and R0)."""
    icons = [child.icon for child in children if not read or child.icon != vc.MISSING or child.rule_id in read]
    return next((icon for icon in WORST_FIRST if icon in icons), vc.OK)


_PANELS = re.compile(r'^[a-m]′?(?:, [a-m]′?)*$')  # '(j)', '(e, h)': panel letters keep their parentheses
_CODE_IN_TEXT = re.compile(r'\s*\((?:[A-Z]{1,2}\d+[a-z]?)\)')  # '(K2)': a rule id inside a sub-check's words


def tidy_checks(lines: Sequence[vc.CheckLine]) -> Tuple[vc.CheckLine, ...]:
    """The Checks block without parentheses inside parentheses (answers and gray numbers, `unnest`), without a rule id
    in a sub-check's words (the id is its tag, D3), 'estimate(s)' as a word and a closing panel letter as 'panel j'."""

    def number_words(text: str) -> str:
        text = re.sub(r'\b1 independent center estimate\(s\)', '1 independent center estimate', text)
        text = re.sub(r'\(([a-m]′?)\)$', r'panel \1', text)  # grey numbers ending '… in (j)': no '(j))'
        return text.replace('estimate(s)', 'estimates')

    out = []
    for line in lines:
        children = tuple(
            replace(child, text=_CODE_IN_TEXT.sub('', child.text), numbers=number_words(unnest(child.numbers, outer=True)))
            for child in line.children
        )
        out.append(replace(line, answer=unnest(line.answer), numbers=number_words(unnest(line.numbers, outer=True)), children=children))
    return tuple(out)


def holding(pills_: Sequence[vc.Pill]) -> List[vc.Pill]:
    return [pill for pill in pills_ if pill.icon in (vc.FAIL, vc.MISSING, vc.THIN)]


def confidence(row: Mapping[str, Any], level: str, missing: Sequence[str] = ()) -> vc.Confidence:
    """The Confidence block: 'not certain because …' from the inputs that hold the level; for no level (uncertain,
    invalid) what is missing (`missing`, the builder's words: the extension, the step)."""
    if not level:
        return vc.Confidence('', '; '.join(missing) or 'the record does not decide', tuple(vc.Pill(vc.MISSING, text) for text in missing))
    found = pills(row, level)
    if level == 'certain':
        return vc.Confidence(level, 'nothing holds it below certain', found)
    return vc.Confidence(level, because_text(found), found)


def because_text(pills_: Sequence[vc.Pill]) -> str:
    """'not certain because …' from the pills that hold the level, each without its gray gloss."""
    because = ', '.join(dict.fromkeys(vc.plain(pill.text).split(' (')[0] for pill in holding(pills_)))
    return f'not certain because {because}' if because else 'not certain'


def chip(verdict, row: Mapping[str, Any], family: str, confidence_block: Optional[vc.Confidence] = None) -> vc.Chip:
    """The action chip: none, glance at the panels of the inputs that hold the level (`GLANCE_PANELS`, decision D5),
    review, extend to a length, resample to a step."""
    action = str(getattr(verdict, 'action', '') or 'none')
    if action == 'extend':
        to = getattr(verdict, 'extend_to_years', None)
        return vc.Chip(vc.CHIP_EXTEND, f'extend to {years(to)}' if finite(to) else 'extend the integration')
    if action == 'resample':
        step = getattr(verdict, 'resample_step_yr', None)
        return vc.Chip(vc.CHIP_RESAMPLE, f'resample to {_number(step)} yr' if finite(step) else 'resample')
    if action == 'spot_check':
        letters = glance_letters(family, confidence_block.pills if confidence_block else (), row)
        return vc.Chip(vc.CHIP_GLANCE, f'glance at {", ".join(f"({x})" for x in letters)}' if letters else 'glance at the figure')
    kind, words = ACTION_WORDS.get(action, (vc.CHIP_NONE, 'no action needed'))
    return vc.Chip(kind, words)


def glance_letters(family: str, pills_: Iterable[vc.Pill], row: Optional[Mapping[str, Any]] = None) -> List[str]:
    """The panels to glance at: those of the inputs that hold the level, in panel order, at most three; a ZLK row's
    gate detail may point an input elsewhere (`GLANCE_OVERRIDES`: an unfitted co-orbital's V)."""
    table = dict(GLANCE_PANELS['long' if family in ('secular', 'zlk') else 'mmr'])
    if family == 'zlk' and row is not None:
        table.update(GLANCE_OVERRIDES.get(text(row.get('zlk_gate_detail')), {}))
    letters: List[str] = []
    for pill in holding(list(pills_)):
        for letter in table.get(pill.key, ()):
            if letter not in letters:
                letters.append(letter)
    return sorted(letters)[:3]


def header(verdict, row: Mapping[str, Any], family: str, confidence_block: vc.Confidence) -> vc.Header:
    status = str(verdict.status)
    return vc.Header(status, str(verdict.confidence or ''), chip(verdict, row, family, confidence_block))


def footer(verdict, row: Mapping[str, Any], docs: str = '') -> Tuple[str, ...]:
    """The raw codes, smallest last: status code, the leaf and the angle rule, reason/origin/veto, the policy's deciding
    step, flags, settings hash, docs."""
    items: List[str] = [f'status {getattr(verdict, "status_code", "")}']
    rule, angle_rule = text(row.get('rec_rule')), text(row.get('geo_angle_rule'))
    if rule:
        items.append(f'leaf {rule}' + (f' · angle rule {angle_rule}' if angle_rule and angle_rule != rule else ''))
    codes: Dict[str, str] = {
        'reason': text(getattr(verdict, 'reason', '')),
        'origin': text(getattr(verdict, 'libration_origin', '')),
        'detail': text(getattr(verdict, 'origin_detail', '')),
        'veto': text(getattr(verdict, 'veto', '')),
        'decided by': text(row.get('pol_decided_by')),
    }
    items += [f'{key} {value}' for key, value in codes.items() if value and value != 'none']
    flags = [str(flag) for flag in (getattr(verdict, 'flags', ()) or ())]
    items.append('flags ' + (', '.join(flags) if flags else 'none'))
    settings_hash = text(row.get('settings_hash'))
    if settings_hash:
        items.append(f'settings {settings_hash[:10]}')
    if docs:
        items.append(f'docs: {docs}')
    return tuple(items)


def text(value: Any) -> str:
    """A cell as text: '' for None and NaN (an empty CSV cell), else stripped."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ''
    return str(value).strip()
