"""The right column of an MMR verdict figure (two-body, three-body, co-orbital 1:1) as a `verdict_column.Column`.

Every block is built from the library of `verdict_text` (the steps `mmr_steps`, the margins `mmr_checks`, the physics
sections, the emphasis lines) and from the summary ⋈ numbers row: no number is computed here, each printed number is a
ledger value or a CSV column (plotting/CLAUDE.md). The Why block is `verdict_column_why.mmr_why`; the header, the
Confidence pills and the footer are `verdict_column_common`'s.

Decisions of the author (2026-11-04, tmp brief D1-D10): numbers are printed where needed (no pointers); the W28 tree's
verdict only in the footer; rule ids only as gray tags on check rows and on the DECISIVE lines; the forced-wave veto
says σ circulates and its short runs are the background's forced wave (D6); the three-body e^q strength is a gray '–'
sub-check the level does not read; terms with plain glosses (induced: borrowed from X; kinematic (forced) libration;
'overruled by').
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from resonances.plotting import combined_numbers as numbers
from resonances.plotting import mmr_verdict_panel as panel
from resonances.plotting import verdict_column as vc
from resonances.plotting import verdict_column_common as common
from resonances.plotting import verdict_column_why as why
from resonances.plotting import verdict_text as vt
from resonances.plotting import verdict_vocab as vv
from resonances.resonance.analysis import policy as pol
from resonances.resonance.analysis.confidence import Cap
from resonances.resonance.analysis.verdict import Flag

FAMILY_TWO_BODY, FAMILY_THREE_BODY, FAMILY_COORBITAL = 'mmr2', 'mmr3', 'coorbital'
DOCS = 'physics/status-tree.md, physics/confidence.md, libration-analysis.md#names'

# --------------------------------------------------------------------------------------------- words

LIBRATING = ('resonant', 'transient', 'near_separatrix')
# the step of the status tree -> its number on the column; the circulation leaves (C5, R0, C, G4c, G4h) are 'C'
STEP_NUMBERS = {'shape': '0', 'sub': '1', 'owner': '2', 'kin': '3', 'drift': '4', 'force': '5', 'zone': '6'}
CIRCULATION_NUMBER = 'C'
QUALITY_NUMBER = 'Q'
POLICY_NUMBER = 'P'
# the margins of steps this branch never asks (a circulation leaf, the strength leaf): one quiet line
OTHER_NUMBER, OTHER_QUESTION, OTHER_ANSWER = '·', 'Also measured', 'by steps this branch does not ask:'
# a verdict whose decision path names no leaf (an analysis built by hand, no rec_rule): one ◆ line says so
UNRECORDED_NUMBER, UNRECORDED_QUESTION = 'R', 'Which rule decided?'
QUALITY_QUESTION = 'Can the data show it?'
# a margin or check id -> the step it nests under (the brief's map; ids not listed go under their ledger step)
NEST = {
    **{key: '0' for key in ('N', 'C1', 'C2', 'C3', 'C4', 'L1', 'L6', 'L7', 'G2', 'G4', 'G5')},
    'S': '1',
    'W': '2',
    **{key: '3' for key in ('L3', 'K1', 'K1e', 'B5', 'B1', 'K2', 'K2b', 'K3s')},  # K3s decides at 3 (37087)
    **{key: '5' for key in ('L4', 'R0', 'F', 'L5', 'O1')},
    **{key: '6' for key in ('L2', 'Z')},
    **{key: QUALITY_NUMBER for key in ('L8', 'L8a', 'L8b', 'L8c', 'C5')},
}
CHECK_ICONS = {
    vt.CheckState.PASS: vc.OK,
    vt.CheckState.THIN: vc.THIN,
    vt.CheckState.FAIL: vc.FAIL,
    vt.CheckState.MISSING: vc.MISSING,
    vt.CheckState.SKIPPED: vc.QUIET,
    vt.CheckState.NOT_APPLICABLE: vc.QUIET,
}
# the policy's deciding step (`pol_decided_by`) as a question and its answer ('{status}' the status word in colour)
# (keyed on `policy.DECIDED_*`, a veto on `DECIDED_VETO` + its name)
POLICY_QUESTIONS = {
    pol.DECIDED_VETO + pol.VETO_FORCED_WAVE: ('Is it the background’s forced wave?', 'Yes → {status}'),
    pol.DECIDED_VETO + pol.VETO_OWNER: ('Does another resonance drive a?', 'Yes → {status}'),
    pol.DECIDED_VETO + pol.VETO_KINEMATIC: ('Is the libration kinematic (forced)?', 'Yes → {status}'),
    pol.DECIDED_VETO + pol.VETO_STRENGTH: ('Is the resonance too weak for these cycles?', 'Yes → {status}'),
    pol.DECIDED_OBJECTION: (
        'Does a measured check object to σ’s own force?',
        'Yes: the shape gives {status}, the objection holds the level',
    ),
    pol.DECIDED_SCALE: ('Can the strength scale overrule the shape?', 'No, an order of magnitude only → {status}'),
    pol.DECIDED_GEOMETRY: ('Is the own force shown?', 'No: the shape gives {status}'),
    pol.DECIDED_DRIFT: ('Does the center drift?', 'Yes → {status}'),
    pol.DECIDED_INSUFFICIENT: ('Is the record long enough?', 'No → {status}'),
    pol.DECIDED_LEG: ('Is the one 1:1 leg a capture?', 'Not shown → {status}'),
    pol.DECIDED_HALF: ('Is the one half-swing the resonance’s own?', 'Not shown → {status}'),
    pol.DECIDED_FORCED_DRIFT: ('Are all counted episodes forced waves?', 'Yes → {status}'),
    pol.DECIDED_MEMBER: ('Does a member of the multiplet librate?', 'Yes → {status}'),
    pol.DECIDED_GATE_CONFIRMED: ('Does the gate confirm the libration?', 'Yes → {status}'),
    pol.DECIDED_GATE_UNAVAILABLE: ('Does the gate give a verdict?', 'No: the shape gives {status}'),
}
# the rule id of a check -> its objection in `pol_objection_check`
OBJECTION_IDS = {
    'L4': pol.OBJECTION_FORCE_LAW,
    'R0': pol.OBJECTION_POTENTIAL,
    'L3': pol.OBJECTION_IDENTITY,
    'F': pol.OBJECTION_STRENGTH,
    'O1': pol.OBJECTION_ONSET,
}
CHECK_NAMES = {
    'L4': 'force law',
    'R0': 'potential',
    'L3': 'identity σ̇ = c (a − a*)',
    'F': 'strength',
    'O1': 'onset test',
    'L5': 'one-pendulum test',
}
# what is missing for an uncertain row, by the extend clause of the policy (`pol_extend_clause`), then by its reason
CLAUSE_MISSING = {
    'A1': 'the 1:1 leg at the record’s end must close',
    'A2': 'the half-swing at the end must close',
    'A': 'the libration at the end must complete its cycles',
    'B': 'the deciding period needs a longer record',
    'C': 'a needed veto needs a longer record',
}
REASON_MISSING = {
    'baseline_needed': 'the record holds too little of the cycle to decide',
    'drift_mechanism_unknown': 'a mechanism for the drift of the center',
    'one_coorbital_leg': 'a closed 1:1 cycle: one leg shows no capture and excludes none',
    'one_half_libration': 'a full cycle: one half-swing shows no capture and excludes none',
    'origin_unknown': 'a measured own force (force law or potential)',
    'undersampled': 'a finer output step',
    'undefined_phase': 'a defined phase of the angle',
}
STATUS_IN_WORDS = {'non_resonant': 'non-resonant', 'near_separatrix': 'near-separatrix'}
_SNAKE = re.compile(r'\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b')
_FLAG_PAREN = re.compile(r'\s*\(([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\)')
FLAG_NAMES = frozenset(str(flag) for flag in Flag) | frozenset(str(cap) for cap in Cap)  # a former cap is a flag now


def words(text: Any) -> str:
    """`text` without codes for the main area: '(flag_name)' of a flag dropped, a status as a word, the ledger's ASCII
    names in symbols (`symbols`), any other snake_case token with spaces ('non_resonant' -> 'non-resonant',
    'insufficient_time' -> 'insufficient time')."""
    text = _FLAG_PAREN.sub(lambda m: '' if m.group(1) in FLAG_NAMES else m.group(0), str(text or ''))
    return _SNAKE.sub(lambda m: STATUS_IN_WORDS.get(m.group(0), vv.snake_words(m.group(0))), symbols(text))


# the ledger's and the old text's ASCII names -> the names the column prints (brief 7: no ASCII variable names)
NAME_SYMBOLS = (
    ('μ_req/μ_th', 'μ needed/μ theory'),
    ('T0_th/T_obs', 'T₀ theory/T₀'),
    ('T0_th F(A)', 'T₀ theory F(A)'),
    ('T0_th', 'T₀ theory'),
    ('φ0_th', 'φ₀ theory'),
    ('μ_req', 'μ needed'),
    ('mu_req', 'μ needed'),
    ('μ_th', 'μ theory'),
    ('mu_th', 'μ theory'),
    ('sigma_dot', 'σ̇'),
    ('a_dot', 'ȧ'),
)
GREEK = {'sigma': 'σ', 'varpi': 'ϖ', 'Omega': 'Ω', 'omega': 'ω', 'lambda': 'λ', 'phi': 'φ', 'Psi': 'Ψ', 'psi': 'ψ'}
_GREEK = re.compile(r'(?<![A-Za-z])(' + '|'.join(sorted(GREEK, key=len, reverse=True)) + r')(?![A-Za-z])')
_SUBSCRIPT = re.compile(r'([σϖΩωλφΨψT])_?(\d)\b')
_COEFFICIENT = re.compile(r'(?<![\w.])(-?\d+)\*([σϖΩωλφΨψ])')
_SIGNED = re.compile(r'(?<=[σϖΩωλφΨψ₀-₉])\s*([+-])\s*(?=\d*[σϖΩωλφΨψ])')
SUBSCRIPTS = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')


def symbols(text: str) -> str:
    """The ASCII names of the ledger and the old text in symbols: 'T0_th' -> 'T₀ theory', 'μ_req' -> 'μ needed',
    'slow sigma follows p varpi' -> 'slow σ follows p ϖ', '8*sigma-1*phi' -> '8σ − φ', 'Omega - varpi_5' -> 'Ω − ϖ₅',
    'ω0' -> 'ω₀'."""
    for old, new in NAME_SYMBOLS:
        text = text.replace(old, new)
    text = _GREEK.sub(lambda m: GREEK[m.group(1)], text)
    text = _SUBSCRIPT.sub(lambda m: m.group(1) + m.group(2).translate(SUBSCRIPTS), text)

    def coefficient(m: re.Match) -> str:
        number = m.group(1)
        return {'1': '', '-1': '−'}.get(number, number.replace('-', '−')) + m.group(2)

    text = _COEFFICIENT.sub(coefficient, text)
    return _SIGNED.sub(lambda m: ' − ' if m.group(1) == '-' else ' + ', text)


_NUMBER_WORDS = {2: 'two', 3: 'three', 4: 'four', 5: 'five'}


def family_of(analysis) -> str:
    """mmr2, mmr3 or coorbital (a 1:1 judged by its own force law)."""
    if analysis.response is not None and analysis.response.force_law == 'coorbital':
        return FAMILY_COORBITAL
    spec = analysis.resonance
    if str(spec.kind) == 'three_body':
        return FAMILY_THREE_BODY
    return FAMILY_COORBITAL if spec.order == 0 and len(spec.planets) == 1 else FAMILY_TWO_BODY


def type_words(spec) -> str:
    """The resonance (`ResonanceSpec`) in words: 'two-body, Jupiter 2:1 (period ratio 1:2), order 1'; 'three-body
    Mars–Jupiter, order 1'; 'co-orbital 1:1 with Earth'. The ratio is that of the key's coefficients, n : n_planet (9N-17
    is Neptune 9:17), so the period ratio P : P_planet is its reverse (17:9, as the outer-belt literature names it)."""
    order = spec.order
    if spec.order == 0 and len(spec.planets) == 1:
        return f'co-orbital 1:1 with {spec.planets[0]}'
    if str(spec.kind) == 'three_body':
        return f'three-body {"–".join(spec.planets)}, order {order}'
    planet, body = abs(spec.planet_coefficients[0]), abs(spec.body_coefficient)
    kind = 'retrograde two-body' if spec.retrograde else 'two-body'
    return f'{kind}, {spec.planets[0]} {planet}:{body} (period ratio {body}:{planet}), order {order}'


# --------------------------------------------------------------------------------------------- subtitle


def _counted(f: vt.MMRFacts) -> int:
    policy = getattr(f.a, 'policy', None)
    ids = getattr(policy, 'counted_episode_ids', None) if policy is not None else None
    return len(ids) if ids is not None else len(f.episodes.counted)


def _end_words(f: vt.MMRFacts) -> str:
    final = common.text(f.v.final_regime)
    if not final or final == str(f.v.angle_status):
        return ''
    if final in ('libration', 'episodic'):
        return '; still librating at the end'
    return f'; ends in {words(final)}'


def shown_words(f: vt.MMRFacts, row: Mapping[str, Any]) -> List[str]:
    """The force checks that pass (`verdict_column_common.force_shown`) in words; a three-body strength is its e^q scale."""
    out = [common.FORCE_WORDS[i] for i in common.force_shown(row)]
    return [('strength scale (e^q)' if word == 'strength' and _strength_scale(f) else word) for word in out]


def _force_clause(f: vt.MMRFacts, row: Mapping[str, Any]) -> str:
    """The geometric subtype by its actual reason (D9), joined to the libration's words: '; own force shown by the
    potential only', ', but two checks object: the force law fails and the energy does not fit the potential' or '; its
    own force is not measured'."""
    shown = shown_words(f, row)
    if shown:
        return f'; own force shown by the {vt.and_list(shown)} only'
    objections = why.objection_checks(row)
    if objections:
        words_ = [common.OBJECTION_WORDS.get(c, vv.snake_words(c)) for c in objections]
        who = 'a check objects' if len(words_) == 1 else f'{_NUMBER_WORDS.get(len(words_), len(words_))} checks object'
        return f', but {who}: {vt.and_list(words_)}'
    return '; its own force is not measured'


def subtype_words(f: vt.MMRFacts, row: Mapping[str, Any]) -> str:
    """The subtype of the verdict in words, by its reason (and origin_detail for a forced wave, D6)."""
    v, reason = f.v, str(f.v.reason)
    n = _counted(f)
    episodes = f'in {vt.plural(n, "episode")}' if n else 'in part of the record'
    if reason == 'dynamic_libration':
        return 'its own libration over the whole record'
    if reason == 'dynamic_episode':
        return f'its own libration {episodes}{_end_words(f)}'
    if reason == 'geometric_libration':
        return f'a libration over the whole record{_force_clause(f, row)}'
    if reason == 'geometric_episode':
        return f'a libration {episodes}{_force_clause(f, row)}'
    if reason == 'kinematic_libration':
        if common.text(v.origin_detail) == 'forced_wave_background':
            return 'σ circulates; its short runs are the background’s forced wave'
        term = vt.term_text(common.text(v.cause_term)) if common.text(v.cause_term) else 'its perihelion term'
        return f'kinematic (forced) libration: bounded by {term}, not this resonance’s force'
    if reason == 'induced_libration':
        owner = common.text(v.owner) or 'another resonance'
        return f'induced: borrowed from {owner}' + ('' if vt.owner_confirmed(f) or not common.text(v.owner) else ' (a candidate owner)')
    if reason == 'baseline_needed':
        rule = vt.shown_rule(f)
        name = vv.rule_name(rule) if rule and rule != f.rule else ''
        return 'too short to decide' + (f': {name}' if name else '')
    simple = {
        'separatrix_zone': 'σ moves near its separatrix',
        'circulation': 'σ circulates',
        'alias': 'σ circulates; the output step folds it (an alias)',
        'no_resonant_potential': 'σ reverses at times, but no potential from a holds it',
        'modulated_circulation': 'σ stalls at the top of its potential without a libration',
        'kinematic_drift': 'σ’s center follows its perihelion term, not this resonance',
        'subresonance_librates': f'a member of the multiplet, {common.text(v.resonant_argument) or "a member"}, librates',
        'drift_mechanism_unknown': 'σ’s center drifts without a known cause',
        'origin_unknown': 'σ librates, but nothing measured shows that this resonance makes it',
        'implausible_strength': 'too weak: the cycles need far more strength than this resonance has',
        'one_coorbital_leg': 'one 1:1 leg: no capture shown, none excluded',
        'one_half_libration': 'one half-swing: no capture shown, none excluded',
        'undersampled': 'the output step is too coarse for the angle',
        'undefined_phase': 'the phase of σ is not defined',
        'unusable_series': 'the series cannot be read',
        'unphysical': 'the orbit is not an ellipse',
    }
    return simple.get(reason, vv.snake_words(reason))


def _decided_words(line: Optional[vc.CheckLine], f: vt.MMRFacts) -> str:
    if line is None:
        return ''
    if line.number == POLICY_NUMBER:
        return f'decided by the policy: {vc.plain(line.question).rstrip("?").lower()}'
    if line.number == UNRECORDED_NUMBER:
        return 'no deciding rule recorded'
    if line.number == CIRCULATION_NUMBER:
        return f'decided by the circulation branch ({vv.rule_name(f.rule, leaf=True) or "circulation"})'
    if line.number == '0':
        rule = vt.shown_rule(f)
        name = vv.rule_name(rule, leaf=rule == f.rule) if rule else ''
        return f'decided at step 0 (angle shape{": " + name if name else ""})'
    short = {value: vt.STEP_SHORT.get(key, key) for key, value in STEP_NUMBERS.items()}
    return f'decided at step {line.number} ({short.get(line.number, "")})'


def subtitle(f: vt.MMRFacts, row: Mapping[str, Any], checks: Sequence[vc.CheckLine]) -> vc.Subtitle:
    key = f.a.resonance.key
    title = f'{key} · {subtype_words(f, row)}'
    decides = next((line for line in checks if line.decides), None)
    detail = [type_words(f.a.resonance), _decided_words(decides, f)]
    shown = shown_words(f, row)
    if shown and str(f.v.status) in LIBRATING:
        detail.append(f'own force shown by the {vt.and_list(shown)}')
    return vc.Subtitle(title, ' · '.join(part for part in detail if part))


# --------------------------------------------------------------------------------------------- key facts


def _cell(row: Mapping[str, Any], column: str) -> str:
    value = row.get(column)
    if numbers.missing(value):
        return ''
    return numbers.format_cell(column, value) + numbers.unit_space(column)


def _subtype_gloss(f: vt.MMRFacts) -> str:
    """The angle subtype in words for the σ fact's gray gloss (`ANGLE_SUBTYPE_WORDS`), '' when it adds nothing; a
    stationary episode whose center moves says both (the header's ', moving center': `decisive_centre_moves`)."""
    subtype = str(f.v.angle_subtype)
    gloss = vv.ANGLE_SUBTYPE_WORDS.get(subtype, '')
    if subtype == 'stationary' and vt.decisive_centre_moves(f):
        gloss = f'{gloss}, center moves'
    return gloss


def _sigma_fact(f: vt.MMRFacts, row: Mapping[str, Any]) -> str:
    # the subtype only, in words and grey ('cycle bands overlap'): the angle status repeats the line's own verb
    gloss = _subtype_gloss(f)
    angle = f' [[{gloss}]]' if gloss else ''
    span = vt.years(row.get('smp_span_yr')) if vt.finite(row.get('smp_span_yr')) else ''
    status = str(f.v.angle_status)
    if f.rule in vt.SINGLE_LEAVES:
        swing = 'one 1:1 leg' if f.rule == 'G4c' else 'one half-swing'
        return f'circulates; judged on {swing} between passages{angle}'
    if status in ('libration', 'episodic') and f.episodes.accepted:
        share = f.episodes.covered_fraction()
        n = _counted(f) or len(f.episodes.accepted)
        head = f'librates in **{vt.percent(share)}** of {span}' if vt.finite(share) and span else 'librates'
        parts = [head]
        if n > 1:
            parts.append(vt.plural(n, 'episode'))
        final = common.text(f.v.final_regime)
        if final:
            parts.append(f'ends in **{final}**')
        return ' · '.join(parts) + angle
    phrase = vt.regime_phrase(status)
    return f'{phrase[2:] if phrase.startswith("σ ") else phrase}{angle}'


def _libration_fact(row: Mapping[str, Any]) -> str:
    parts = []
    if vt.finite(row.get('ep_period_med_yr')):
        parts.append(f'P {vt.years(row["ep_period_med_yr"])}')
    if vt.finite(row.get('ep_half_amp_med_deg')):
        amp = f'half-amplitude {vt.fmt(row["ep_half_amp_med_deg"])}°'
        if vt.finite(row.get('ep_half_amp_max_deg')):
            amp += f' (max {vt.fmt(row["ep_half_amp_max_deg"])}°)'
        parts.append(amp)
    centre = why.centre_text(row.get('ep_centre_mod_2pi_deg'))
    if centre:
        parts.append(f'center {centre}')
    return ' · '.join(parts)


def _episode_facts(f: vt.MMRFacts) -> List[vc.Fact]:
    accepted = f.episodes.accepted
    if len(accepted) < 2:
        return []
    decisive = f.episodes.decisive
    out = []
    for i, episode in enumerate(accepted):
        mark = ' ◆' if decisive is not None and vt._same_id(episode.get('episode_id'), decisive.get('episode_id')) else ''
        counted = '' if _episode_counted(f, episode) else ' (not counted)'
        text = (
            f'#{numbers.format_cell("episode_id", episode.get("episode_id"))} '
            f'{vt.interval(episode.get("t_start_yr"), episode.get("t_end_yr"))} P {vt.years(episode.get("period_med_yr"))} '
            f'A {vt.fmt(episode.get("half_amp_med_deg"))}° ({vt.fmt(episode.get("half_amp_max_deg"))}°){mark}{counted}'
        )
        out.append(vc.Fact('episodes' if i == 0 else '', text))
    return out


def _episode_counted(f: vt.MMRFacts, episode: Mapping[str, Any]) -> bool:
    policy = getattr(f.a, 'policy', None)
    ids = getattr(policy, 'counted_episode_ids', None) if policy is not None else None
    if ids is None:
        return any(row is episode for row in f.episodes.counted)
    return any(vt._same_id(episode.get('episode_id'), i) for i in ids)


def _resonance_fact(f: vt.MMRFacts, row: Mapping[str, Any]) -> vc.Fact:
    if str(f.v.reason) == 'induced_libration' and common.text(f.v.owner):
        # an owner librates and makes the episodes; a driver circulates and forces a wave (response.md#roles)
        role = 'driver' if common.text(f.v.cause_role) == 'driver' else 'owner'
        return vc.Fact(role, f'{f.v.owner}' + ('' if vt.owner_confirmed(f) else ' (candidate)'))
    parts = []
    for column, label in (('ctx_a_res_nominal_au', 'a_res'), ('ctx_a_mean_au', 'mean a')):
        cell = _cell(row, column)
        if cell:
            parts.append(f'{label} {cell}')
    if vt.finite(row.get('ctx_order')):
        parts.append(f'order {vt.fmt(row["ctx_order"])}')
    return vc.Fact('resonance', ' · '.join(parts)) if parts else vc.Fact('resonance', f.a.resonance.key)


def _range(row: Mapping[str, Any], low: str, high: str, unit: str = '') -> str:
    """'lo–hi unit' of two columns, one number when both print the same; '' when either is missing."""
    if not (vt.finite(row.get(low)) and vt.finite(row.get(high))):
        return ''
    lo, hi = vt.fmt(row[low]), vt.fmt(row[high])
    return (lo if lo == hi else f'{lo}–{hi}') + unit


def _orbit_fact(row: Mapping[str, Any]) -> str:
    parts = []
    for label, low, high, unit in (('e', 'ctx_e_min', 'ctx_e_max', ''), ('i', 'ctx_inc_min_deg', 'ctx_inc_max_deg', '°')):
        span = _range(row, low, high, unit)
        if span:
            parts.append(f'{label} {span}')
    for column, label in (('ctx_q_min_au', 'q_min'), ('ctx_Q_max_au', 'Q_max')):
        if vt.finite(row.get(column)):
            parts.append(f'{label} {vt.fmt(row[column])} au')
    return ' · '.join(parts)


def _overlap_fact(row: Mapping[str, Any]) -> str:
    line = vt.orbit_line(row).rstrip('.')
    for head in ('Radial overlap with ', 'Radial overlap: '):
        if line.startswith(head):
            return line[len(head) :]
    return line


def _run_fact(row: Mapping[str, Any]) -> str:
    parts = []
    if vt.finite(row.get('smp_span_yr')):
        parts.append(vt.years(row['smp_span_yr']))
    if vt.finite(row.get('smp_step_yr')):
        parts.append(f'output {vt.fmt(row["smp_step_yr"])} yr')
    if vt.finite(row.get('flt_cutoff_yr')):
        parts.append(f'cutoff {vt.years(row["flt_cutoff_yr"])}')  # '4.61 kyr' beside 'run 300 kyr', not '4 608 yr'
    return ' · '.join(parts)


def key_facts(f: vt.MMRFacts, row: Mapping[str, Any]) -> Tuple[vc.Fact, ...]:
    facts = [vc.Fact('σ', _sigma_fact(f, row))]
    libration = _libration_fact(row) if str(f.v.angle_status) in ('libration', 'episodic', 'near_separatrix') else ''
    if libration:
        facts.append(vc.Fact('libration', libration))
    facts += _episode_facts(f)
    facts.append(_resonance_fact(f, row))
    for label, value in (('orbit', _orbit_fact(row)), ('overlaps', _overlap_fact(row)), ('run', _run_fact(row))):
        if value:
            facts.append(vc.Fact(label, value))
    return tuple(facts)


# --------------------------------------------------------------------------------------------- checks


def _strength_scale(f: vt.MMRFacts) -> bool:
    """The three-body (or retrograde) strength is the e^q scale only (`Strength.method` eq_scale)."""
    strength = f.a.strength
    return strength is not None and str(strength.method) == 'eq_scale'


def _power(value: float) -> str:
    exponent = math.log10(value) if value > 0 else float('nan')
    if math.isfinite(exponent) and abs(exponent - round(exponent)) < 1e-9 and round(exponent) >= 2:
        digits = str(int(round(exponent))).translate(str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'))
        return f'10{digits}'
    return vt.fmt(value)


def scale_words(f: vt.MMRFacts, row: Mapping[str, Any]) -> str:
    """The words of the three-body e^q strength sub-check, its ratio (str_ratio) inside: a scale only, never a veto
    below ×10³ after the small divisors (MMR_STRENGTH_IMPLAUSIBLE_RATIO)."""
    ratio = row.get('str_ratio')
    bound = _power(float(f.s.mmr_strength_implausible_ratio))
    scale = f'×{vt.sci(ratio)}; ' if vt.finite(ratio) else ''
    return f'strength, order of magnitude: scale only ({scale}an objection only from ×{bound} after small divisors)'


# the operator a passing comparison prints first in a margin's detail (verdict_text `MARGIN_DETAILS`, `_l8_rows`; `compare`
# prints the operator as the comparison came out, so another one there is a failed comparison)
PASS_OPS = {'L5': '≤', 'L6': '≤', 'L7': '≤', 'L8a': '<', 'L8b': '≥', 'O1': '<', 'C1': '≥', 'C4': '≥', 'B5': '≥'}
_FIRST_NUMBER = re.compile(r'−?-?\d[\d.]*(?:e-?\d+)?')
_OPERATOR_AFTER = re.compile(r'(?:°| P| rev)?\s([≤≥<>])\s')


def _printed_fails(key: str, detail: str) -> bool:
    """Whether the detail's first number is compared with the operator of a failed comparison ('shift/amplitude 5.77 >
    0.5' for L6); False when the first number is compared with nothing ('turn per step 365°, not graded: …')."""
    first = _FIRST_NUMBER.search(detail)
    after = _OPERATOR_AFTER.match(detail[first.end() :]) if first else None
    return key in PASS_OPS and after is not None and after.group(1) != PASS_OPS[key]


def failed_reading(f: vt.MMRFacts, key: str, icon: str, detail: str) -> bool:
    """Whether a margin's row shows a failed or marginal measurement, so that its name must say so (`MMR_MARGIN_FAILED`):
    the comparison it prints failed (any icon: a – row too, 2897 O1), or by its state where the detail prints no
    comparison: L1 fewer turning points than the full bar, L2 a cycle near the separatrix (thin: in the zone or the
    amplitude above its bar), L3/L4/R0 failed, Z only estimated, O1 thin (a wave before ≥ the thin bar), W in the thin
    band, F outside its band."""
    if _printed_fails(key, detail):
        return True
    if icon not in (vc.THIN, vc.FAIL):
        return False
    if key == 'L1':
        margin = f.margins.get('L1')
        return margin is not None and bool(vt.holds(margin.value, '<', f.s.mmr_l1_tp_full))
    if key in ('L2', 'Z', 'O1'):
        return True
    if key in ('L3', 'L4', 'R0'):
        return icon == vc.FAIL
    if key == 'W':
        return 'thin band' in detail
    if key == 'F':
        return ' outside [' in detail
    return False


def _reading_name(f: vt.MMRFacts, key: str, name: str, icon: str, detail: str) -> str:
    """The sub-check's name as its numbers read: the failed form (`MMR_MARGIN_FAILED`) when `failed_reading`, else
    `name` (a passing condition)."""
    if not failed_reading(f, key, icon, detail):
        return name
    failed = vv.MMR_MARGIN_FAILED.get(f'{key}:coorbital' if f.coorbital else key) or vv.MMR_MARGIN_FAILED.get(key)
    return words(failed) if failed else name


K2_ENTRY = '3 kinematic (slow centre)'  # the ledger family of K2's conditions (decide `K2_SHARE`, `K2_CORR`, …)


def _sub_check(f: vt.MMRFacts, row: Mapping[str, Any], check: vt.CheckRow) -> vc.SubCheck:
    icon = CHECK_ICONS.get(check.state, vc.QUIET)
    if check.state == vt.CheckState.SKIPPED and check.counts == vt.COUNTS_NOT_MEASURED and 'not needed' not in check.note:
        icon = vc.MISSING  # needed but not measured (the force law of a short episode), not a test skipped by design
    name = re.sub(r'^\d+\w*\s+', '', check.short_name or check.name)
    if check.key.startswith('U:') and vt.ledger_words(check.key[2:]) == check.key[2:]:
        # a ledger entry without words of its own ('3 K1 kinematic: slow sigma follows p varpi', 'K3s the centre
        # circulates only by slips'): named by its rule (`RULE_NAMES`), the entry's own words in symbols after it
        rule = _rule_id(check)
        if rule in vv.RULE_NAMES:
            head = re.sub(r'^\d*\w*\s+(?:' + re.escape(rule) + r':?\s+)?', '', check.key[2:]).split(':')[0].strip()
            tail = f': {symbols(head)}' if len(head.split()) > 2 else ''  # 'the centre circulates only by slips'
            name = f'{vv.RULE_NAMES[rule]} near its convention{tail}'
    if check.key.startswith(f'U:{K2_ENTRY}: '):
        # one condition of K2 (decide `slow_centre_is_kinematic`): the entry's own words, not K2's conclusion (79653:
        # 'a does not answer sigma, force law R2 (h1)' 0.382 < 0.4 is the force law, not the slow centre's correlation)
        test = re.sub(r'\s*\([^()]*\)$', '', check.key[len(f'U:{K2_ENTRY}: ') :]).replace('R2', 'R²')
        name = f'{vv.RULE_NAMES["K2"]}, one of its tests near its convention: {symbols(test)}'
    detail = check.detail.strip()
    if detail.startswith('(') and detail.endswith(')'):
        detail = detail[1:-1]
    detail = '; '.join(part for part in (detail, check.note) if part)
    if check.printed_id == 'C5' and str(f.v.angle_subtype) == 'alias' and ' a-scales from a*;' in detail:
        # the alias leaf's margin (decide `_c5_margin`): full at the far-from-resonance bound MMR_FAR_FROM_RES_FACTOR,
        # while the leaf itself needs only MMR_C5_FAR_RATIO_MIN (1999 UQ: 66.2 against 10 and 3); both bars named
        bar = vt.fmt(f.s.mmr_c5_far_ratio_min)
        detail = detail.replace(' a-scales from a*;', f' a-scales from a* for a full margin, the leaf needs ≥ {bar};', 1)
        name = 'margin of the alias leaf: far from a*, one branch'
    if check.printed_id == 'W':
        # one R² against two bars: the step accepts an owner from MMR_OWNER_R2_MIN, the margin is full below
        # MMR_W_R2_NONE (1921 2J-1: 0.059 < 0.7 at step 2, < 0.4 here); each bar named where it is printed
        none = f'< {vt.fmt(f.s.mmr_w_r2_none)}'
        detail = re.sub(rf'({re.escape(none)})(?=\)|;|$)', r'\1, too low even for a partial fit', detail, count=1)
    if check.printed_id == 'F' and _strength_scale(f) and pol.OBJECTION_STRENGTH_SCALE not in why.objection_checks(row):
        text = scale_words(f, row)  # its words carry the ratio; the margin's [0.1, 10] is not this check's bar
        return vc.SubCheck(vc.QUIET, text, '', 'F')
    if not check.key.startswith('U:'):
        name = _reading_name(f, check.printed_id, name, icon, detail)
    return vc.SubCheck(icon, words(name), words(detail), _rule_id(check))


def _rule_id(check: vt.CheckRow) -> str:
    """The id on the sub-check's tag: a near-threshold ledger entry ('U:<name>') by the rule its name reads."""
    if check.key.startswith(f'U:{K2_ENTRY}:'):
        return 'K2'
    if check.key.startswith('U:'):
        found = vt.entry_rule_id(check.key[2:])
        return found if found in vv.RULE_NAMES else 'U'
    return check.printed_id


def _nest_number(check: vt.CheckRow) -> str:
    rule_id = check.printed_id
    if check.key.startswith('U:'):
        entry = check.key[2:]
        rule_id = vt.entry_rule_id(entry)
        if rule_id not in NEST and entry[:1].isdigit():
            return entry[:1]
    return NEST.get(rule_id, NEST.get(check.key, '0'))


def _split_answer(outcome: str) -> Tuple[str, str]:
    """An outcome as (answer, numbers): a closing parenthesis at the end moves to the gray numbers."""
    text = words(outcome).strip()
    if not text.endswith(')'):
        return text, ''
    depth = 0
    for i in range(len(text) - 1, -1, -1):
        depth += {')': 1, '(': -1}.get(text[i], 0)
        if depth == 0:
            head = text[:i].rstrip()
            return (head, text[i + 1 : -1]) if head else (text, '')
    return text, ''


def _policy_decided(row: Mapping[str, Any]) -> str:
    decided = common.text(row.get('pol_decided_by'))
    return '' if decided in ('', 'tree') else decided


def _policy_line(f: vt.MMRFacts, decided: str) -> vc.CheckLine:
    step = decided.split(';')[0]
    question, answer = POLICY_QUESTIONS.get(step, ('What does the policy decide?', '→ {status}'))
    status = f'!!{vt.status_head(f.v.status).lower()}!!'
    numbers_ = ''
    if step == pol.DECIDED_OBJECTION:
        numbers_ = '; '.join(common.OBJECTION_WORDS.get(c, c) for c in why.objection_checks(f.row))
    return vc.CheckLine(vc.DECIDES, POLICY_NUMBER, question, answer.format(status=status), numbers_, POLICY_NUMBER, decides=True)


def _step_line(f, step: vt.Step, children: Tuple[vc.SubCheck, ...], decided: str) -> vc.CheckLine:
    number = STEP_NUMBERS.get(step.key, CIRCULATION_NUMBER)
    if step.key == 'rest':
        names = step.title.rsplit(':', 1)[0].replace(', ', ' · ')
        answer = '— not reached' + ('; measured anyway:' if children else '')
        return vc.CheckLine(vc.QUIET, step.number, names, answer, children=children, quiet=True)
    question = step.question or step.title
    answer, numbers_ = _split_answer(step.outcome)
    if step.key == 'owner' and numbers_:  # the acceptance bar of an owner, apart from the W margin's level
        bar = f'< {vt.fmt(f.s.mmr_owner_r2_min)}'
        numbers_ = re.sub(rf'({re.escape(bar)})$', r'\1 needed to accept an owner', numbers_)
    rule_id = ','.join(step.ids)
    if step.state == vt.StepState.LEAF:
        if not decided:
            return vc.CheckLine(vc.DECIDES, number, question, answer, numbers_, vt.shown_rule(f), children, decides=True)
        overruled = f'overruled: the policy decides at step {POLICY_NUMBER}' + (
            ' (its veto not upheld)' if pol.NOT_UPHELD in decided else ''
        )
        icon = vc.FAIL if pol.NOT_UPHELD in decided else _overruled_icon(f, step, children)
        return vc.CheckLine(icon, number, question, f'{answer}; {overruled}', numbers_, vt.shown_rule(f), children)
    if step.state in (vt.StepState.NOT_NEEDED, vt.StepState.NOT_APPLICABLE, vt.StepState.NOT_REACHED):
        return vc.CheckLine(vc.QUIET, number, question, answer, numbers_, rule_id, children, quiet=True)
    if step.state == vt.StepState.NOT_RUN:
        if answer.startswith('not triggered') and not children:
            # the sibling scan runs only on a trigger (a moving centre, separate cycle bands): none fired, so the question
            # does not arise, a step not needed (–), not a gap (?) on a resonant/certain row (1921 2J-1)
            return vc.CheckLine(vc.QUIET, number, question, answer, numbers_, rule_id, children, quiet=True)
        return vc.CheckLine(vc.MISSING, number, question, answer, numbers_, rule_id, children)
    icon = common.worst(children, step.ids) if children else vc.OK
    return vc.CheckLine(icon, number, question, _unread_failure(answer, icon, step, children), numbers_, rule_id, children)


# the checks of the own-force step (5) by their summary columns: an overruled leaf there shows its own result
FORCE_STEP_CHECKS = ('check_mmr_a_response', 'check_mmr_potential')


def _overruled_icon(f: vt.MMRFacts, step: vt.Step, children: Sequence[vc.SubCheck]) -> str:
    """The icon of a leaf the policy overruled, by its own result (the legend's – is 'not needed', and this step was
    needed): at the own-force step ✗ when the force law or the potential fails, ? when neither was measured; elsewhere
    the worst of its sub-checks that is not ✓, else –."""
    if step.key == 'force':
        results = [common.text(f.row.get(column)) for column in FORCE_STEP_CHECKS]
        if 'fail' in results:
            return vc.FAIL
        if not any(result == 'pass' for result in results):
            return vc.MISSING
        return vc.QUIET
    worst = common.worst(children, step.ids) if children else vc.OK
    return worst if worst != vc.OK else vc.QUIET


def _unread_failure(answer: str, icon: str, step: vt.Step, children: Sequence[vc.SubCheck]) -> str:
    """The answer of a ✗ step whose ✗ comes from a sub-check the answer does not read (523899: 'B1 not measured' with
    the identity failing below): the failure is named, so the icon is not left unexplained."""
    if icon != vc.FAIL or any(child.icon == vc.FAIL and child.rule_id in step.ids for child in children):
        return answer
    named = [vc.plain(child.text).split(':')[0] for child in children if child.icon == vc.FAIL]
    return f'{answer}; but {vt.and_list(named)} (below)' if named else answer


def _quality_line(f: vt.MMRFacts, row: Mapping[str, Any], children: List[vc.SubCheck]) -> vc.CheckLine:
    flags = {str(flag) for flag in f.v.flags}
    if 'filter_sensitive' in flags:
        changed = ', '.join(q for q in common.text(row.get('flt_sensitive_quantities')).split(';') if q)
        children.append(vc.SubCheck(vc.THIN, 'filter-sensitive: a deciding number changes at cutoff ×2', words(changed)))
    if 'short_period_aliased' in flags:
        children.append(vc.SubCheck(vc.THIN, 'the output step folds short periods'))
    worst = common.worst(children) if children else vc.OK
    reserved = [vc.plain(c.text).split(':')[0] for c in children if c.icon in (vc.FAIL, vc.MISSING, vc.THIN)]
    # the reservations in their failed words (`_reading_name`): 'Partly: the output step is too coarse for σ', never
    # 'Yes, but the output step resolves σ'; a failed sub-check makes it 'No'
    answer = 'Yes.' if worst == vc.OK else f'{"No" if worst == vc.FAIL else "Partly"}: {"; ".join(reserved[:3])}.'
    return vc.CheckLine(worst, QUALITY_NUMBER, QUALITY_QUESTION, answer, words(_run_fact(row)), children=tuple(children))


def _after_the_steps(lines, steps, placed, leftover, decided: str) -> List[vc.CheckLine]:
    """The quiet line after the tree's steps: decided at step 0 (insufficient time: 385458 9N-17 G5w) steps 1-6 as one
    'not reached' line, as after any other leaf, with the margins those steps measured under it; else, on a branch
    without the tree's later steps, the margins of steps it does not ask ('Also measured')."""
    shape_leaf = any(line.decides and line.number == '0' for line in lines)
    if shape_leaf and not decided and not placed & set(STEP_NUMBERS.values()) - {'0'}:
        answer = '— not reached' + ('; measured anyway:' if leftover else '')
        names = ' · '.join(vt.STEP_SHORT[key] for key in STEP_NUMBERS if key != 'shape')
        return [vc.CheckLine(vc.QUIET, '1–6', names, answer, children=leftover, quiet=True)]
    if leftover and not any(step.key == 'rest' for step in steps):
        return [vc.CheckLine(vc.QUIET, OTHER_NUMBER, OTHER_QUESTION, OTHER_ANSWER, children=leftover, quiet=True)]
    return []


def checks(
    f: vt.MMRFacts, text: vt.VerdictText, row: Mapping[str, Any], emphasis: Sequence[vt.Emphasis] = ()
) -> Tuple[Tuple[vc.CheckLine, ...], str]:
    """(the Checks block, the rule of its ◆ line): Q first, then the steps of the tree with their margins nested
    (`NEST`), and the policy's line when the policy, not the tree, decided (`pol_decided_by`). A failed check that no
    margin row shows (a CONFLICT line of `emphasis`: the identity of 523899, 142694) gets its own ✗ row under its step."""
    nested: Dict[str, List[vc.SubCheck]] = {}
    for check in text.checks:
        if check.is_cap:
            continue
        nested.setdefault(_nest_number(check), []).append(_sub_check(f, row, check))
    shown_ids = {child.rule_id for children in nested.values() for child in children}
    for line in emphasis:
        if line.kind == vt.EMPHASIS_CONFLICT and line.rule_id in NEST and line.rule_id not in shown_ids:
            numbers_ = ', '.join(f'{_label(p.column)} {p.text}'.strip() if p.labelled else p.text for p in line.parts)
            name = _reading_name(f, line.rule_id, words(line.name), vc.FAIL, numbers_)
            nested.setdefault(NEST[line.rule_id], []).append(vc.SubCheck(vc.FAIL, name, words(numbers_), line.rule_id))
            shown_ids.add(line.rule_id)
    decided = _policy_decided(row)
    lines = [_quality_line(f, row, nested.pop(QUALITY_NUMBER, []))]
    steps = list(text.steps)
    placed = {STEP_NUMBERS.get(step.key, CIRCULATION_NUMBER) for step in steps if step.key != 'rest'}
    for step in steps:
        number = STEP_NUMBERS.get(step.key, CIRCULATION_NUMBER)
        if step.key == 'rest':  # the margins of steps not reached hang under the one quiet line
            orphans = tuple(child for key in sorted(nested) if key not in placed for child in nested[key])
            lines.append(_step_line(f, step, orphans, decided))
            continue
        lines.append(_step_line(f, step, tuple(nested.get(number, ())), decided))
    leftover = tuple(child for key in sorted(nested) if key not in placed for child in nested[key])
    lines += _after_the_steps(lines, steps, placed, leftover, decided)
    if decided:
        lines.append(_policy_line(f, decided))
    if not any(line.decides for line in lines):  # the contract's one ◆, without inventing a rule
        answer = f'none recorded in the decision path → !!{vt.status_head(f.v.status).lower()}!!'
        lines.append(vc.CheckLine(vc.DECIDES, UNRECORDED_NUMBER, UNRECORDED_QUESTION, answer, decides=True))
    rule = POLICY_NUMBER if decided else vt.shown_rule(f)
    return tuple(lines), rule


# --------------------------------------------------------------------------------------------- confidence


def confidence(f: vt.MMRFacts, row: Mapping[str, Any]) -> vc.Confidence:
    level = str(f.v.confidence or '')
    if level in ('', 'none'):
        missing = []
        clause = common.text(row.get('pol_extend_clause'))
        shape = _shape_missing(f) if clause in ('', 'A') else ''
        if shape:  # an insufficient-time angle: what its own shape still needs, not 'the libration at the end'
            missing.append(shape)
        elif clause in CLAUSE_MISSING:
            missing.append(CLAUSE_MISSING[clause])
        reason = REASON_MISSING.get(str(f.v.reason))
        if reason and reason not in missing and not (shape and str(f.v.reason) == 'baseline_needed'):
            missing.append(reason)
        return _plain_confidence(common.confidence(row, '', missing))
    block = _plain_confidence(common.confidence(row, level))
    pills = tuple(_mmr_pill(f, row, pill) for pill in block.pills)
    because = words(common.because_text(pills)) if level != 'certain' else block.because
    return vc.Confidence(block.level, because, pills)


def _mmr_pill(f: vt.MMRFacts, row: Mapping[str, Any], pill: vc.Pill) -> vc.Pill:
    """A pill in the MMR's own words: the owner veto (`_veto_pill`), V not measured by the check it lacks, S by the
    parts it reads (`policy._mmr_stable`: L1 full and no filter_sensitive)."""
    if pill.key == 'veto':
        return _veto_pill(f, pill)
    state = {name: state for name, state, _, _ in common.parse_inputs(row.get('pol_level_inputs'))}.get(pill.key, '')
    text = pill.text
    if pill.key == 'V' and state == 'not_measured' and common.text(row.get('check_mmr_identity')) not in ('pass', 'fail', 'not_applicable'):
        text = text.replace('a needed alternative not measured', 'the identity σ̇ = c (a − a*) not measured', 1)
    if pill.key == 'F' and _strength_scale(f):  # a three-body strength is its order-of-magnitude scale only
        text = re.sub(r'\bstrength(?=[,)])', 'strength scale e^q', text)
    if pill.key == 'S':
        text = text.replace(common.STATE_WORDS['S'].get(state, common.STATE_WORDS['S']['']), _stable_words(f, state), 1)
    return vc.Pill(pill.icon, text, pill.key, pill.ceiling)


def _stable_words(f: vt.MMRFacts, state: str) -> str:
    """S of an MMR row in words: what holds at cutoff ×½ and ×2 (the decisive episode's turning points, L1) and at ×2
    (no deciding number changes: no filter_sensitive), or which part fails."""
    if state == 'full':
        return 'filter-stable [[enough turning points at cutoff ×½, 1 and 2; no deciding number changes at ×2]]'
    parts = []
    l1 = f.margins.get('L1')
    if l1 is not None and vt.holds(l1.value, '<', f.s.mmr_l1_tp_full):
        parts.append('too few turning points at some cutoff')
    if 'filter_sensitive' in {str(flag) for flag in f.v.flags}:
        parts.append('a deciding number changes at cutoff ×2')
    return 'filter-sensitive' + (f' [[{"; ".join(parts)}]]' if parts else '')


def _veto_pill(f: vt.MMRFacts, pill: vc.Pill) -> vc.Pill:
    """The veto's pill says what was measured: 'driven by 5E-16+0+11: measured' for an owner the decision kept (a
    candidate: 'driven by X: a candidate'), 'overruled by <the veto in words>' for the others; a thin veto says so."""
    if pill.key != 'veto' or str(f.v.veto) != 'owner' or not common.text(f.v.owner):
        return pill
    measured = 'measured' if vt.owner_confirmed(f) else 'a candidate'
    thin = common.THIN_VETO if pill.text.endswith(common.THIN_VETO) else ''
    return vc.Pill(pill.icon, f'driven by {f.v.owner}: {measured}{thin}', pill.key, pill.ceiling)


# what an insufficient-time angle still needs, by its angle subtype (`AngleSubtype`; G5w/G5a: the slow swing)
SHAPE_MISSING = {
    'arc': 'the arc must turn back and close a cycle',
    'stall': 'σ must leave its stall and close a cycle',
    'few_cycles': 'more turning points: at least {turning_points} to close one cycle',
}


def _shape_missing(f: vt.MMRFacts) -> str:
    """The missing item of an insufficient-time row in the words of its angle subtype: 'the slow swing must close: 3
    slow periods, a record of 300 kyr' (G5w, G5a: MMR_EXTEND_SLOW_CYCLES and the extension's length), 'the arc must
    turn back and close a cycle', …; '' for any other angle."""
    if str(f.v.angle_status) != 'insufficient_time' or f.rule in vt.SINGLE_LEAVES:
        return ''
    to = f.v.extend_to_years if str(f.v.action) == 'extend' else None
    length = f', a record of {vt.years(to)}' if vt.finite(to) else ''
    rule = f.angle_rule or vt.insufficient_angle_rule(f)
    if rule in ('G5w', 'G5a') or vt.slow_swing_decided(f):
        return f'the slow swing must close: {vt.fmt(f.s.mmr_extend_slow_cycles)} slow periods{length}'
    words_ = SHAPE_MISSING.get(str(f.v.angle_subtype))
    if words_ is None:
        return ''
    return words_.format(turning_points=f.s.libration_min_turning_points) + length


def _plain_confidence(block: vc.Confidence) -> vc.Confidence:
    """The block with its words free of codes ('overruled by forced_wave' -> 'overruled by forced wave')."""
    pills = tuple(vc.Pill(pill.icon, words(pill.text), pill.key, pill.ceiling) for pill in block.pills)
    return vc.Confidence(block.level, words(block.because), pills)


# --------------------------------------------------------------------------------------------- evidence


def _flat(text: str) -> str:
    """A gloss that goes into gray parentheses without nested ones (`verdict_column_common.unnest`, outer: `[[…]]` adds
    the outer pair): 'x (y)' -> 'x, y'; a formula's parentheses ('c (a − a*)') and panel letters stay."""
    return common.unnest(text, outer=True)


def _keeps_parentheses(inner: str, after: str, before: str) -> bool:
    """Whether a parenthesis of an Evidence value belongs to the statement, not to its gray numbers: a formula's own
    ('(T₀ theory/T₀)²', '= (1/8)φ', 'c (a − a*)', the same test as `verdict_column_common.unnest`) or panel letters."""
    if after.isalnum() or (after and after in '²/') or before.strip() == '=':
        return True
    return common._formula(inner) or bool(common._PANELS.match(inner))


def _evidence_title(section: vt.PhysicsSection, key: str) -> str:
    if section.title.startswith('Is σ the own pendulum'):
        head = f'Evidence: is this {key}’s own pendulum?'
    elif section.title.startswith('What makes the cycles'):
        head = 'Evidence: what makes the cycles?'
    else:
        head = f'Evidence: {section.title}'
    return head + (f' [[{_flat(words(section.subtitle))}]]' if section.subtitle else '')


EVIDENCE_HEAD_MAX = 64  # characters of an Evidence value before its numbers go grey (one line of the sub-column)


def _closed(numbers_: str) -> str:
    """Numbers that do not end in ']' (the markup '[[…]]' would close early on ']]]'): the parts reordered so that one
    without a closing bracket comes last, else the interval named a band."""
    if not numbers_.endswith(']'):
        return numbers_
    parts = numbers_.split('; ')
    open_ = [part for part in parts if not part.endswith(']')]
    if open_:
        return '; '.join([part for part in parts if part.endswith(']')] + open_)
    return re.sub(r'\[([^\[\]]+), ([^\[\]]+)\]$', r'band \1–\2', numbers_)


def evidence_value(text: str) -> str:
    """An Evidence value: the statement short, in symbols (`symbols`), its numbers gray: the parenthesized details move
    to '[[…]]'; a statement still longer than EVIDENCE_HEAD_MAX keeps what comes before its first ': ' and the rest goes
    gray too ('μ needed 0.000256 against μ theory 7.6e-5–0.000105 [[ratio …]]')."""
    text = words(text).strip()
    grey, head, depth, piece, start = [], [], 0, [], 0
    for i, ch in enumerate(text):
        if ch == '(':
            depth += 1
            if depth == 1:
                piece, start = [], i
                continue
        elif ch == ')' and depth:
            depth -= 1
            if depth == 0:
                after, before = text[i + 1 : i + 2], text[max(0, start - 2) : start]
                if _keeps_parentheses(''.join(piece), after, before):
                    head += ['(', *piece, ')']
                else:
                    grey.append(_flat(''.join(piece)).strip())
                continue
        (piece if depth else head).append(ch)
    head_text = re.sub(r'\s+', ' ', ''.join(head)).strip().rstrip(':;,').replace(' ,', ',').replace(' ;', ';').replace(' :', ':')
    if len(head_text) > EVIDENCE_HEAD_MAX and ': ' in head_text:
        head_text, rest = head_text.split(': ', 1)
        head_text = head_text.strip()
        grey.insert(0, _flat(rest.strip()))
    grey = [_tidy_grey(part) for part in grey if part]
    return head_text + (f' [[{_closed("; ".join(grey))}]]' if grey else '')


def _tidy_grey(part: str) -> str:
    """A gray part without a sentence's stop before ';' ('7.11 kyr.; ±11.5°') and without the owner theory's scale said
    twice ('scale 1.04–1.26 …; fitted/predicted 1.04–1.26, |Δφ₀| …': the same ratio, verdict_text `_owner_theory`)."""
    part = re.sub(r'(?<=\w)\.(?=;|$)', '', part)
    scale = re.search(r'\bscale (\S+–\S+)', part)
    if scale:
        part = re.sub(r'fitted/predicted ' + re.escape(scale.group(1)) + r'(?:,\s*|;\s*|$)', '', part).rstrip('; ,')
    return part


SLOW_SWING_RULES = ('G5w', 'G5a')


def _slow_swing_evidence(f: vt.MMRFacts, row: Mapping[str, Any]) -> Optional[vc.Evidence]:
    """Evidence A of a row decided by the slow swing (G5w): its period against T₀ theory F(A), its swing in a against the
    pendulum's, the slow identity, from the ledger entries of G5w and the geo_slow_swing_* columns; None for G5a (one
    slow arc: nothing is compared, the evidence is quiet)."""
    path = [entry for entry in f.v.decision_path if entry.name.startswith('G5w ') and entry.result is not None]
    if not path:
        return None

    def entries(word: str):
        return [entry for entry in path if word in entry.name]

    def icon(found) -> str:
        return vc.OK if found and all(entry.result for entry in found) else vc.FAIL

    def band(found) -> str:
        bounds = sorted(entry.threshold for entry in found)
        return f'band {vt.fmt(bounds[0])}–{vt.fmt(bounds[-1])}' if len(bounds) == 2 else ''

    rows = []
    period = entries('period 2 (t2 - t1)')
    if period:
        theory = row.get('geo_slow_swing_t0_theory_yr')
        numbers_ = [f'T₀ theory {vt.years(theory)}' if vt.finite(theory) else '', band(period)]
        value = f'slow swing {vt.years(row.get("geo_slow_swing_period_yr"))} = {vt.fmt(period[0].value)} × T₀ theory F(A)'
        rows.append(vc.EvidenceRow(icon(period), 'own period', f'{value} [[{", ".join(n for n in numbers_ if n)}]]'))
    swing = entries('a swing between the turns')
    if swing:
        measured, theory = row.get('geo_slow_swing_a_au'), row.get('geo_slow_swing_a_theory_au')
        numbers_ = [f'{vt.sci(measured)} au against {vt.sci(theory)} au' if vt.finite(measured) and vt.finite(theory) else '', band(swing)]
        value = f'a swings {vt.fmt(swing[0].value)} × the pendulum’s'
        rows.append(vc.EvidenceRow(icon(swing), 'swing in a', f'{value} [[{", ".join(n for n in numbers_ if n)}]]'))
    identity = entries('slow identity')
    if identity:
        compared = ', '.join(
            f'{"|slope/c − 1|" if "slope" in entry.name else "corr"} {vt.compare(entry.value, str(entry.op), entry.threshold)}'
            for entry in identity
        )
        held = all(entry.result for entry in identity)
        rows.append(
            vc.EvidenceRow(
                icon(identity), 'σ̇ follows a', f'{"the slow identity holds" if held else "the slow identity fails"} [[{compared}]]'
            )
        )
    title = f'Evidence: is the slow swing {f.a.resonance.key}’s own pendulum? [[half of one slow swing, at the slow scale]]'
    return vc.Evidence(title, tuple(rows)) if rows else None


def evidence(f: vt.MMRFacts, text: vt.VerdictText, row: Mapping[str, Any]) -> Tuple[vc.Evidence, ...]:
    """The physics sections (A 'is σ the own pendulum of X?', B 'what makes the cycles?') as icon, label, value (short,
    numbers gray: `evidence_value`); an owner search that accepted no candidate is clean evidence (✓ 'owner search'),
    the three-body e^q strength gray. A row decided by its slow swing (G5w, G5a: insufficient time) speaks of that swing
    in A (`_slow_swing_evidence`), never of the cycles of a rejected run."""
    out = []
    owner = f.a.owner
    slow = str(f.v.angle_status) == 'insufficient_time' and (f.angle_rule or vt.insufficient_angle_rule(f)) in SLOW_SWING_RULES
    for section in text.physics:
        if slow and section.title.startswith('Is σ the own pendulum'):
            replaced = _slow_swing_evidence(f, row)
            if replaced is not None:
                out.append(replaced)
            continue
        rows = []
        for item in section.rows:
            icon = item.icon if item.icon in (vc.OK, vc.FAIL, vc.MISSING) else vc.QUIET
            label = item.label
            if item.key == 'owner' and owner is not None and owner.accepted is None:
                icon = vc.OK if str(owner.check) in ('pass', 'fail') else vc.QUIET  # searched and none found: clean
                label = 'owner search'
            if item.key == 'scale' and pol.OBJECTION_STRENGTH_SCALE not in why.objection_checks(row):
                icon, label = vc.QUIET, 'strength (scale)'
            rows.append(vc.EvidenceRow(icon, label, evidence_value(item.text)))
        if rows:
            out.append(vc.Evidence(_evidence_title(section, f.a.resonance.key), tuple(rows)))
    return tuple(out)


# --------------------------------------------------------------------------------------------- caveats


def slots(row: Mapping[str, Any]) -> Dict[str, str]:
    """The slots of the `verdict_vocab.FLAG_CAVEATS` sentences from the row's columns ({jump}, {ratio}, {exit_P},
    {entry_P}); a slot without a value is left out, and so is the sentence that needs it (`flag_caveat`)."""
    out = {}
    if vt.finite(row.get('regime_start_yr')):
        out['jump'] = vt.years(row['regime_start_yr'])
    if vt.finite(row.get('smp_da_sp_over_a_excursion')):
        out['ratio'] = vt.fmt(row['smp_da_sp_over_a_excursion'])
    # the distance of a suspected edge from the record's edge, in the episode's periods (`geometry.EdgeEvent`)
    if why.true(row.get('exit_suspected')) and vt.finite(row.get('exit_remaining_P')):
        out['exit_P'] = vt.fmt(row['exit_remaining_P'])
    for kind in ('entry', 'record_entry'):
        if 'entry_P' not in out and why.true(row.get(f'{kind}_suspected')) and vt.finite(row.get(f'{kind}_remaining_P')):
            out['entry_P'] = vt.fmt(row[f'{kind}_remaining_P'])
    return out


def flag_caveat(flag: str, status: str, filled: Mapping[str, str]) -> str:
    """The `verdict_vocab.FLAG_CAVEATS` sentence of one flag on this status with its slots filled (`slots`); '' when the
    flag has none, it does not apply to the status, or a slot it needs has no value."""
    entry = vv.FLAG_CAVEATS.get(flag)
    if entry is None or status not in entry[0]:
        return ''
    try:
        return entry[1].format(**filled)
    except KeyError:
        return ''


def flag_caveats(f: vt.MMRFacts, row: Mapping[str, Any]) -> List[str]:
    """The sentences of `verdict_vocab.FLAG_CAVEATS` for the row's flags and status (a member caveat only when the status
    rests on a member, rule 1); a sentence whose slot has no value is not said."""
    status, filled, out = str(f.v.status), slots(row), []
    for flag in (str(flag) for flag in f.v.flags):
        if flag in vv.MEMBER_CAVEATS and f.rule != '1':
            continue
        if flag == 'no_full_physics_support' and 'strength_scale_only' in {str(x) for x in f.v.flags}:
            continue  # the strength-scale caveat already names the one support (2348, 24820)
        sentence = flag_caveat(flag, status, filled)
        if sentence and sentence not in out:
            out.append(sentence)
    return out


def _failed(text: vt.VerdictText, emphasis: Sequence[vt.Emphasis]) -> Dict[str, str]:
    """id -> the numbers of every failed check the column shows: the failed margins and the conflict lines."""
    out: Dict[str, str] = {}
    for check in text.checks:
        if check.state == vt.CheckState.FAIL and not check.is_cap:
            out.setdefault(check.printed_id, check.detail.strip('()'))
    for line in emphasis:
        if line.kind == vt.EMPHASIS_CONFLICT:
            out.setdefault(line.rule_id, ', '.join(f'{_label(p.column)} {p.text}'.strip() for p in line.parts))
    return out


def unread_caveats(f: vt.MMRFacts, row: Mapping[str, Any], text: vt.VerdictText, emphasis: Sequence[vt.Emphasis]) -> List[str]:
    """'The {check} fails ({numbers}); the level does not read it, because …' for a failed check that is no objection
    the level reads (`pol_objection_check`), on a librating or uncertain row; the e^q strength scale explained."""
    status = str(f.v.status)
    if status not in LIBRATING + ('uncertain',):
        return []
    read = set(why.objection_checks(row))
    shown = shown_words(f, row)
    out = []
    for rule_id, detail in _failed(text, emphasis).items():
        objection = OBJECTION_IDS.get(rule_id, '')
        if rule_id == 'F' and _strength_scale(f):
            if pol.OBJECTION_STRENGTH_SCALE in read:
                continue
            ratio = row.get('str_ratio')
            needs = f'these cycles need ×{vt.sci(ratio)} of it' if vt.finite(ratio) else 'it does not fit these cycles'
            bound = _power(float(f.s.mmr_strength_implausible_ratio))
            out.append(
                f'The three-body strength is known only as the scale μ_i μ_j e^q and {needs}; the coefficient is '
                f'unknown, so the level does not read it (an objection only from ×{bound} after small divisors).'
            )
            continue
        if objection and objection in read:
            continue
        name = CHECK_NAMES.get(rule_id)
        if name is None:
            continue
        if status == 'uncertain':
            because = 'the row has no level, and the status rests on another step'
        elif shown:
            because = f'the own force is shown by the {vt.and_list(shown)}'
        else:
            because = 'the policy reads only the objections it names'
        numbers_ = f' [[{words(detail)}]]' if detail else ''
        out.append(f'The {name} fails{numbers_}; the level does not read it, because {because}.')
    return out


# --------------------------------------------------------------------------------------------- numbers


def _label(column: str) -> str:
    if not column:
        return ''
    return numbers.column_words(column) or vv.snake_words(column.split('_', 1)[-1])


EMPHASIS_HEADS = {
    vt.EMPHASIS_DECISIVE: '',
    vt.EMPHASIS_LIMIT: 'limits the level: ',
    vt.EMPHASIS_UNMEASURED: 'not measured: ',
}


def decisive_line(line: vt.Emphasis, read: Sequence[str]) -> str:
    """One DECISIVE line in words with its id: 'energy fits the potential [R0]: …'; the parts labeled in words."""
    head = EMPHASIS_HEADS.get(line.kind, '')
    if line.kind == vt.EMPHASIS_CONFLICT:
        objection = OBJECTION_IDS.get(line.rule_id, '')
        head = 'fails, read as an objection: ' if objection and objection in read else 'fails, not read: '
    parts = ', '.join((f'{_label(p.column)} {p.text}' if p.labelled and p.column else p.text) for p in line.parts)
    note = line.note
    if line.kind == vt.EMPHASIS_CONFLICT and note == vt.OBJECTION_NOTE:
        note = ''
    if line.kind == vt.EMPHASIS_UNMEASURED and note.startswith('not measured'):
        note = note[len('not measured') :].strip()
        if note.startswith('(') and note.endswith(')'):
            note = note[1:-1]
    body = ': '.join(part for part in (parts, note) if part)
    return f'{head}{line.name} [{line.rule_id}]' + (f': {body}' if body else '')


# the groups of Numbers (brief 9), most important first; a group with no value is left out
GROUPS = (
    (
        'LIBRATION',
        (
            'ep_centre_mod_2pi_deg',
            'ep_period_med_yr',
            'ep_half_amp_med_deg',
            'ep_half_amp_max_deg',
            'ep_n_tp',
            'ep_centre_shift_over_amp',
            'ep_coverage_start_gap_P',
            'ep_coverage_end_gap_P',
            'geo_n_tp',
            'geo_period_cycle_yr',
            'geo_centre_max_pass_rev',
            'geo_drift_fraction',
            'geo_n_revolutions',
            'geo_n_slips',
        ),
    ),
    (
        'RESPONSE OF a',
        (
            'rsp_b1_ratio',
            'rsp_identity_slope_ratio',
            'rsp_identity_corr',
            'rsp_force_window_yr',
            'rsp_force_n_windows',
            'rsp_force_R2_h1',
            'rsp_force_R2_h2',
            'rsp_force_phi0_deg',
            'rsp_phi0_max_dev_deg',
            'rsp_phi0_tol_deg',
            'rsp_coorb_K_over_K_theory',
            'rsp_coorb_energy_R2',
            'rsp_omega0_ratio',
            'rsp_b5_ratio',
        ),
    ),
    (
        'PENDULUM, ENERGY',
        (
            'ep_T0_min_kyr',
            'ep_T0_max_kyr',
            'ep_half_width_au',
            'ep_mu_eff',
            'sep_h_med',
            'sep_zone_method',
            'sep_zone_h_lo',
            'sep_zone_h_hi',
            'sep_VE_R2',
            'sep_period_law_med',
            'sep_VE_VF_agree_fraction',
            'sep_mechanism',
        ),
    ),
    (
        'RESONANCE',
        (
            'ctx_order',
            'ctx_a_res_nominal_au',
            'ctx_a_star_au',
            'ctx_a_mean_au',
            'ctx_delta_a_au',
            'ctx_c_rad_yr_au',
            'ctx_rate_expected_deg_yr',
            'ctx_rate_measured_deg_yr',
        ),
    ),
)
LATER_GROUPS = (
    (
        'KINEMATICS',
        (
            'kin_corr_rate',
            'kin_e_free',
            'kin_e_forced',
            'kin_modulation_period_kyr',
            'kin_two_omega_period_yr',
            'kin_pred_swing_deg',
            'kin_meas_swing_deg',
        ),
    ),
    ('SLOW SCALE', ('slow_period_yr', 'slow_halfamp_deg', 'slow_cutoff_yr', 'slow_secular_period_yr', 'slow_cause_level')),
    ('SUBRESONANCES, DRIVER', ('sub_n_scanned', 'sub_best', 'sub_accepted', 'own_name', 'own_n_candidates')),
    (
        'STEP, FILTER',
        (
            'smp_span_yr',
            'smp_step_yr',
            'smp_step_rotation_deg_max',
            'smp_far_ratio',
            'smp_da_sp_over_a_excursion',
            'flt_cutoff_yr',
            'flt_gain_at_pcyc',
            'rsp_response_cutoff_yr',
        ),
    ),
)


def _group_rows(row: Mapping[str, Any], columns: Sequence[str], grey: Mapping[str, str]) -> Tuple[Tuple[str, str], ...]:
    """(label, value) of the columns with a value and a label in words; a min/max pair of RANGE_LABELS one line."""
    out: List[Tuple[str, str]] = []
    done = set()
    for column in columns:
        if column in done or numbers.missing(row.get(column)):
            continue
        pair = next(((lo, hi) for lo, hi in numbers.RANGE_LABELS if column in (lo, hi) and lo in columns and hi in columns), None)
        if pair is not None and not numbers.missing(row.get(pair[0])) and not numbers.missing(row.get(pair[1])):
            lo, hi = (numbers.format_cell(c, row.get(c)) for c in pair)
            value = (lo if lo == hi else f'{lo}–{hi}') + numbers.unit_space(pair[0])
            out.append((numbers.RANGE_LABELS[pair], value))
            done.update(pair)
            continue
        label = numbers.column_words(column)
        if not label:
            continue
        value = numbers.cell_with_unit(column, row.get(column))
        out.append((label, value + (' (not used)' if column in grey else '')))
        done.add(column)
    return tuple(out)


def _orbit_group(row: Mapping[str, Any]) -> Optional[vc.NumberGroup]:
    rows = []
    for label, columns, unit in (
        ('e', ('ctx_e_min', 'ctx_e_med', 'ctx_e_max'), ''),
        ('i', ('ctx_inc_min_deg', 'ctx_inc_med_deg', 'ctx_inc_max_deg'), '°'),
    ):
        if all(not numbers.missing(row.get(c)) for c in columns):
            rows.append((label, ' / '.join(numbers.format_cell(c, row[c]) for c in columns) + unit))
    if not numbers.missing(row.get('ctx_q_min_au')) and not numbers.missing(row.get('ctx_Q_max_au')):
        rows.append(('q_min / Q_max', f'{numbers.format_cell("ctx_q_min_au", row["ctx_q_min_au"])} / ' + _cell(row, 'ctx_Q_max_au')))
    if not numbers.missing(row.get('ctx_P_orb_yr')):
        rows.append(('orbital period', _cell(row, 'ctx_P_orb_yr')))
    return vc.NumberGroup('ORBIT (min / med / max)', tuple(rows)) if rows else None


def _episode_group(f: vt.MMRFacts, episodes: Sequence[Mapping[str, Any]]) -> Optional[vc.NumberGroup]:
    if not episodes:
        return None
    lines = panel.explained_episode_lines(episodes, f.row.get('ep_episode_id'))
    title = lines[0].group if lines else 'EPISODES'
    rows = tuple(('', words(re.sub(r'^\*', '◆', line.value))) for line in lines)
    return vc.NumberGroup(title.replace('TP', 'turning points'), rows)


def number_groups(f: vt.MMRFacts, row: Mapping[str, Any], episodes, constants, text: vt.VerdictText) -> Tuple[vc.NumberGroup, ...]:
    out: List[vc.NumberGroup] = []
    episode = _episode_group(f, episodes)
    if episode is not None:
        out.append(episode)
    grey = dict(text.grey_columns)
    for title, columns in GROUPS:
        rows = _group_rows(row, columns, grey)
        if rows:
            name = title
            if title == 'LIBRATION' and not numbers.missing(row.get('ep_episode_id')):
                name = f'LIBRATION (episode #{numbers.format_cell("episode_id", row.get("ep_episode_id"))})'
            out.append(vc.NumberGroup(name, rows))
    orbit = _orbit_group(row)
    if orbit is not None:
        out.append(orbit)
    periods = panel.period_lines(row, constants)
    if periods:
        out.append(vc.NumberGroup('REFERENCE PERIODS', tuple((line.label, line.value) for line in periods)))
    for title, columns in LATER_GROUPS:
        rows = _later_rows(row, title, columns, grey)
        if rows:
            out.append(vc.NumberGroup(title, rows))
    return tuple(out)


def _later_rows(row: Mapping[str, Any], title: str, columns: Sequence[str], grey: Mapping[str, str]) -> Tuple[Tuple[str, str], ...]:
    """The rows of a LATER_GROUPS group: KINEMATICS with its forced-e model marked (`_kinematics_rows`), the
    subresonances with their ranked members."""
    rows = _group_rows(row, columns, grey)
    if title == 'KINEMATICS':
        rows = _kinematics_rows(row, rows)
    if title.startswith('SUBRESONANCES'):
        rows += tuple(('', line.value) for line in panel.ranked_members_lines(row, title))
    return rows


FORCED_E_COLUMNS = ('kin_e_free', 'kin_e_forced')


def _kinematics_rows(row: Mapping[str, Any], rows: Tuple[Tuple[str, str], ...]) -> Tuple[Tuple[str, str], ...]:
    """KINEMATICS with its free and forced e marked when either is ≥ 1 (1997 GK3, a planet-crossing orbit: 1.86 and
    2.13 for e 0.26–0.30): an e of 1 or more is no ellipse, so the linear forced-e model fails there, and so does the
    half-swing it predicts."""
    if not any(vt.holds(row.get(column), '>=', 1) for column in FORCED_E_COLUMNS):
        return rows
    marked = {numbers.column_words(column): ' (≥ 1: the linear forced-e model fails here)' for column in FORCED_E_COLUMNS}
    marked[numbers.column_words('kin_pred_swing_deg')] = ' (from that model)'
    return tuple((label, value + marked.get(label, '')) for label, value in rows)


POLICY_VETO_ENTRY = 'policy:veto '  # the decision-path entries of the policy's own veto (`policy._mmr_veto_reading`)


def policy_lines(f: vt.MMRFacts, decided: str) -> List[str]:
    """The DECISIVE lines of a veto the policy set (`pol_decided_by` 'veto:<name>'): its ledger comparisons that held,
    named by the policy line's question, with the id P."""
    step = decided.split(';')[0]
    if not step.startswith(pol.DECIDED_VETO):
        return []
    name = step[len(pol.DECIDED_VETO) :]
    question = POLICY_QUESTIONS.get(step, ('', ''))[0].rstrip('?')
    out = []
    for entry in f.v.decision_path:
        if not entry.name.startswith(f'{POLICY_VETO_ENTRY}{name}') or not entry.result:
            continue
        comparison = vt.compare(entry.value, str(entry.op), entry.threshold)
        label = _label(entry.column) if entry.column else entry.name.split(':', 2)[-1].strip()
        out.append(f'{(question or name).lower()} [{POLICY_NUMBER}]: {label} {comparison}')
    return out


def numbers_block(f, row, episodes, constants, text, emphasis) -> vc.Numbers:
    read = why.objection_checks(row)
    decided = _policy_decided(row)
    tree = [_objection_read(f, line, _potential_read(f, line, decisive_line(line, read), read), read) for line in emphasis]
    if pol.NOT_UPHELD in decided:  # the tree's leaf the policy did not uphold: its numbers stay, named so
        tree = [f'not upheld: {line}' if line.split(' [')[0] in _decisive_names(emphasis) else line for line in tree]
    decisive = tuple(symbols(line) for line in policy_lines(f, decided) + tree)
    return vc.Numbers(decisive, number_groups(f, row, episodes, constants, text))


def _potential_read(f: vt.MMRFacts, line: vt.Emphasis, text: str, read: Sequence[str]) -> str:
    """An R0 line as the tree read it: a 1:1 energy integral whose R² passes but whose scale fails, or an R0 the tree
    read failing while the filter guard emptied the published check (2016 JA 5 kyr, leaf 5a: the policy's objection
    'potential' is that reading), says so; any other line is `text`."""
    if line.rule_id != 'R0' or line.kind != vt.EMPHASIS_DECISIVE:
        return text
    scale = vt.coorbital_energy_scale_clause(f)
    if not (scale or vt.tree_read_potential_fail(f)):
        return text
    head = 'fails, read as an objection: ' if 'potential' in read else 'fails, not read: '
    return head + text + (f', but {scale}' if scale else '')


# the summary column of each check an objection names (`OBJECTION_IDS`)
OBJECTION_CHECK_COLUMNS = {
    'L4': 'check_mmr_a_response',
    'R0': 'check_mmr_potential',
    'L3': 'check_mmr_identity',
    'F': 'check_mmr_strength',
    'O1': 'check_mmr_onset',
}


def _objection_read(f: vt.MMRFacts, line: vt.Emphasis, text: str, read: Sequence[str]) -> str:
    """A DECISIVE line of a check that fails and is read as an objection says so first, as a conflict line does (2016
    JA: 'a obeys the force law [L4]: … 0.355 < 0.7' under an objection the policy read)."""
    column = OBJECTION_CHECK_COLUMNS.get(line.rule_id, '')
    if line.kind != vt.EMPHASIS_DECISIVE or text.startswith('fails') or not column:
        return text
    if OBJECTION_IDS[line.rule_id] not in read or common.text(f.row.get(column)) != 'fail':
        return text
    return f'fails, read as an objection: {text}'


def _decisive_names(emphasis: Sequence[vt.Emphasis]) -> Tuple[str, ...]:
    return tuple(line.name for line in emphasis if line.kind == vt.EMPHASIS_DECISIVE)


# --------------------------------------------------------------------------------------------- the column


def _why(f: vt.MMRFacts, row: Mapping[str, Any]) -> vc.Why:
    """The Why block (`verdict_column_why.mmr_why`)."""
    return why.mmr_why(f, row)


def mmr_column(
    analysis,
    settings,
    row: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    constants: Sequence[Mapping[str, Any]] = (),
) -> vc.Column:
    """The right column of one MMR verdict. `row` = summary ⋈ numbers ({**numbers_row, **summary_row}), `episodes` =
    episodes.csv rows of the angle, `constants` = constants.csv rows (the reference periods)."""
    family = family_of(analysis)
    text = vt.explain_mmr(analysis, settings, row, episodes)
    f = vt.MMRFacts(analysis, settings, row, episodes)
    # D1: a number is printed wherever it is needed (the physics block's ones too)
    emphasis = vt.number_emphasis(f, text.steps, text.checks, vt.info_items(f, text.physics))
    conf = confidence(f, row)
    header = common.header(f.v, row, family, conf)
    lines, rule = checks(f, text, row, emphasis)
    lines = common.tidy_checks(lines)
    caveats = flag_caveats(f, row) + unread_caveats(f, row, text, emphasis)
    return vc.Column(
        family=family,
        header=header,
        subtitle=subtitle(f, row, lines),
        key_facts=key_facts(f, row),
        why=_why(f, row),
        confidence=conf,
        checks=lines,
        evidence=evidence(f, text, row),
        caveats=tuple(caveats),
        numbers=numbers_block(f, row, episodes, constants, text, emphasis),
        footer=common.footer(f.v, row, DOCS),
        decided_rule=rule,
    )


def figure_column(analysis, rows=None, settings=None) -> vc.Column:
    """The column the MMR verdict figure draws, exactly as `mmr_combined.draw_combined` builds it: `rows` = (summary,
    numbers, episodes, constants) as read from the CSV files, by default the analysis' own (`mmr_combined.figure_rows`);
    `settings` default: from the configuration. `verdict_column.column_text` writes it as text."""
    if rows is None:
        from resonances.plotting.mmr_combined import figure_rows

        rows = figure_rows(analysis)
    summary, numbers_row, episodes, constants = rows
    settings = panel.config_settings(settings)
    return mmr_column(analysis, settings, {**numbers_row, **summary}, episodes, constants)
