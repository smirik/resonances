"""The Why block of the right column (`verdict_column.Why`, brief block 4; the author's decisions D1, D9 of 2026-11-04;
the wording of the second review, 2026-11-05: the author's 1921 mockup, every test said by what it means physically,
with its number): at most five plain sentences from deterministic templates, then one last line.

The sentences, in order:

    1 what is seen       the shape of σ (ω, the proper angle σ), where, how much, with the panel letters <<…>>
    2 what it means      whose force makes the cycles (the force law, the energy, the strength, each with its number),
                         or why they are not this resonance's own
    3 what is excluded   positive sentences with their numbers, only what was measured (the sibling scan, the owner
                         search, a carries the swing); an alternative whose margin is thin is 'not fully ruled out'
    4 what limits        'Nothing limits the confidence.' at certain, else 'The confidence stops at X: …' from the
                         inputs that hold it (`policy.level_inputs`)

An optional sentence (the three-body strength scale) is dropped first when a row has more than five.

The templates are keyed on the family (MMR, secular/ZLK) and the reason (`MMR_TEMPLATES`, `LONG_TEMPLATES`); inside a
template the veto, the origin detail, the deciding leaf and the angle word choose the words. Every number is a column
of the row or a ledger value, printed through `verdict_text`'s formatters; the physics stays in `verdict_text`'s
predicates (`owner_confirmed`, `secular_reference`, `single_swing_clause`, `zlk_physics_line`, …). A three-body
strength is an order-of-magnitude scale: it is never said to make the cycles.

The last line comes only from what the code computed: the policy's extend clause (`pol_extend_clause` A1/A2/A/B/C)
with `extend_to_years`, the veto a longer record measures (`pol_veto_need_yr`), the baseline (`base_required_kyr`,
`base_period_yr`, MMR_EXTEND_SLOW_CYCLES), an unresolved 'both resonances at once', the free node's separation time;
'' when nothing is computed. Its heads: 'Needed: …' (uncertain, invalid), 'Would change the verdict: …' (only what
would), 'Next check: …' (an extension that tests, without a promise), and 'A longer record cannot make it resonant: …'
for a transient row with a circulation between its episodes (or before its first one).

The panel letters are those of docs/reading-figures.md: MMR (a) angle, (b) mod 360°, (c) rate, (d) a, (e) response of
a, (h) energy per cycle, (i) center, (j) related centers, (k)/(l) portraits, (m) FAIR; secular/ZLK (a) angle, (b) mod
360°, (c) rate, (d) a, (e)/(f) e and i, (g) the rate of the element / the free ω drift, (h) center, (i) portrait, (j)
the free e vector or σ against its rate.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from resonances.plotting import verdict_column as vc
from resonances.plotting import verdict_column_common as vcc
from resonances.plotting import verdict_text as vt
from resonances.plotting.verdict_vocab import BASE_CAUSE_WORDS
from resonances.resonance.analysis import policy as pol
from resonances.resonance.analysis.mmr.analyse import QS_STALL_ENTRY
from resonances.resonance.analysis.verdict import Reason

MAX_SENTENCES = 5
WOULD_CHANGE = 'Would change the verdict: '
NEEDED = 'Needed: '
NEXT_CHECK = 'Next check: '
NO_CHANGE = 'A longer record cannot make it resonant: '
NOTHING_LIMITS = 'Nothing limits the confidence.'
OPTIONAL = '\x00'  # the mark of a sentence dropped first when a row has more than MAX_SENTENCES (`_fit`)

# MMR panels (docs/reading-figures.md, the combined MMR figure)
M_ANGLE, M_MOD, M_RATE, M_AXIS, M_RESPONSE, M_ENERGY = 'a', 'b', 'c', 'd', 'e', 'h'
M_CENTRE, M_RELATED, M_PORTRAIT, M_EPISODE_PORTRAIT = 'i', 'j', 'k', 'l'
# secular and ZLK panels (docs/reading-figures.md, the secular and Lidov-Kozai figure)
L_ANGLE, L_MOD, L_E, L_I, L_DRIFT, L_CENTRE, L_VECTOR = 'a', 'b', 'e', 'f', 'g', 'h', 'j'

# the force checks that showed the force (`verdict_column_common.force_shown`) in words, with their panels; the
# numbers are added where the row has them (`_pendulum_words`, `_strength_sentence`)
FORCE_SEEN = {
    'L4': ('ȧ follows its force law', (M_RESPONSE,)),
    'R0': ('the cycle energy fits the potential built from a', (M_ENERGY,)),
    'F': ('the strength of its averaged term can make cycles of this period', ()),
}
COORBITAL_FORCE_SEEN = 'ȧ follows the 1:1 force law of the planet’s own mass'
# the averaged strength (two-body, retrograde harmonic); any other method (eq_scale) is an order-of-magnitude scale
DIRECT_STRENGTH = ('direct_averaging', 'direct_harmonic')
# a veto in words, for the limit phrase and the meaning (the code stays in the footer)
VETO_WORDS = vcc.VETO_WORDS
# the free-ω gate's no-verdict outcomes in words (`secular.decide`; docs/physics/secular-zlk.md#zlk-free-omega): (the
# short words of a Checks row or pill, `verdict_column_long`; the Why's sentence)
GATE_NO_VERDICT = {
    'short_window': (
        'too few libration periods for the free-element split',
        'the record holds too few libration periods for the free-element split',
    ),
    'undefined_phase': (
        'the free vector passes too near the origin',
        'the free eccentricity vector passes too near the origin for its phase to be followed',
    ),
    'planetary_residual': (
        'a residual rate matches a planetary one',
        'a rate of the free elements matches a planetary one too closely to be told apart',
    ),
    'unavailable': ('the free elements could not be built', 'the free elements could not be built'),
    'disabled': ('the gate is switched off', 'the gate is switched off'),
}


# --------------------------------------------------------------------------------------------- shared


def _panels(*letters: str) -> str:
    return vc.panels(letters)


def _sentence(text: str) -> str:
    """The text as one sentence: a capital first letter (markup and σ/ω kept) and a full stop."""
    text = text.strip()
    if not text:
        return ''
    if text[0].isalpha() and text[0].islower() and text[0] not in 'σωϖȧ':
        text = text[0].upper() + text[1:]
    return text if text.endswith(('.', '?', '!')) else text + '.'


def _join(items: Sequence[str]) -> str:
    items = [item for item in items if item]
    if len(items) <= 1:
        return ''.join(items)
    return ', '.join(items[:-1]) + ' and ' + items[-1]


def true(value: Any) -> bool:
    """A flag cell as a bool: True, 'True', '1' or '1.0' (a CSV cell read back)."""
    return value is True or vcc.text(value).lower() in ('true', '1', '1.0')


def centre_text(value: Any) -> str:
    """A libration center as the column prints it, in [0°, 360°): '0.693°', '270°', '0°' just below 360° (359.7° would
    round to 360°); '' when not measured. Key facts and the Why of both families print it so."""
    if not vt.finite(value):
        return ''
    value = float(value) % 360
    return f'{vt.fmt(0.0 if value > 359.5 else value)}°'


def objection_checks(row: Mapping[str, Any]) -> Tuple[str, ...]:
    """The checks whose measured objection the level reads (`pol_objection_check`, 'force_law;potential')."""
    return tuple(name for name in vcc.text(row.get('pol_objection_check')).split(';') if name)


def longest_gap(rows: Sequence[Mapping[str, Any]]) -> str:
    """'during 825–927 kyr': the longest gap between two consecutive episodes of `rows` (sorted by start), where the
    angle does not librate; '' without one."""
    gaps = [(vt._float(a.get('t_end_yr')), vt._float(b.get('t_start_yr'))) for a, b in zip(rows, rows[1:])]
    gaps = [(t0, t1) for t0, t1 in gaps if vt.finite(t0) and vt.finite(t1) and t1 > t0]
    if not gaps:
        return ''
    t0, t1 = max(gaps, key=lambda gap: gap[1] - gap[0])
    return f'during {vt.interval(t0, t1)}'


def _span(row: Mapping[str, Any]) -> str:
    """'4.81–32.5 kyr' (`verdict_text.interval`)."""
    return vt.interval(row.get('t_start_yr'), row.get('t_end_yr'))


def _share(fraction: Any, span_yr: Any = None) -> str:
    """', 65% of the record' (', 65% of the 100 kyr record' with the record's span; 'under 1%' below half a percent;
    '' when not measured)."""
    if not vt.finite(fraction):
        return ''
    record = f'the {vt.years(span_yr)} record' if vt.finite(span_yr) else 'the record'
    return f', {"under 1%" if float(fraction) < 0.005 else vt.percent(fraction)} of {record}'


_ONE_UNIT = re.compile(r'^(\S+–\S+) (yr|kyr|Myr)$')


def _spans(rows: Sequence[Mapping[str, Any]], limit: int = 3) -> str:
    """'2.39–10.1, 26.1–33.6 and 51.1–70.7 kyr' (one unit at the end when every span has it), up to `limit` spans,
    then 'and N more'."""
    spans = [_span(row) for row in rows[:limit]]
    more = len(rows) - limit
    found = [_ONE_UNIT.match(span) for span in spans]
    if len(spans) > 1 and more <= 0 and all(found) and len({match.group(2) for match in found}) == 1:
        return f'{_join([match.group(1) for match in found])} {found[0].group(2)}'
    return f'{", ".join(spans)} and {"one" if more == 1 else more} more' if more > 0 else _join(spans)


def _fit(sentences: Sequence[str]) -> Tuple[str, ...]:
    """At most MAX_SENTENCES sentences: the optional ones (marked OPTIONAL) go first, then the sentences before the last
    (the limit sentence stays); duplicates once."""
    items = list(dict.fromkeys(s for s in sentences if s and s.strip(OPTIONAL)))
    while len(items) > MAX_SENTENCES and any(s.startswith(OPTIONAL) for s in items):
        items.remove(next(s for s in reversed(items) if s.startswith(OPTIONAL)))
    items = [_sentence(s.lstrip(OPTIONAL)) for s in items]
    if len(items) > MAX_SENTENCES:
        items = items[: MAX_SENTENCES - 1] + items[-1:]
    return tuple(dict.fromkeys(items))


def _inputs(row: Mapping[str, Any]) -> Dict[str, Tuple[str, str, str]]:
    """name -> (state, detail, ceiling) of the policy's level inputs (`pol_level_inputs`)."""
    return {name: (state, detail, ceiling) for name, state, detail, ceiling in vcc.parse_inputs(row.get('pol_level_inputs'))}


# the inputs that hold the level, in the Why's words where the pill's are a label ('close to the separatrix')
LIMIT_STATE_WORDS = {
    ('M', 'thin'): 'the cycles lie close to the separatrix',
    ('M', 'not_measured'): 'the separatrix margin is not measured',
    ('S', 'thin'): 'the result depends on the filter cutoff',
    ('S', 'not_measured'): 'the filter stability is not measured',
    ('V', 'partial'): 'some alternatives are not fully ruled out',
    ('V', 'not_measured'): 'a needed alternative is not measured',
    ('F', 'none'): 'no test measured its own force',
    ('F', 'not_shown'): 'no test measured its own force',
    ('second_capture', 'unresolved'): 'both resonances may act at once',
}
# a veto whose words depend on the reason (the kinematic veto of a drift is no bound: 62243)
VETO_BY_REASON = {('kinematic', 'kinematic_drift'): 'the kinematic drift'}


def _objection_names(row: Mapping[str, Any]) -> List[str]:
    """The objection the level reads (`pol_objection_check`) in the Why's words (`OBJECTION_SEEN`, else the pill's)."""
    return [
        OBJECTION_SEEN[name][0] if name in OBJECTION_SEEN else vcc.OBJECTION_WORDS.get(name, vt.snake_words(name))
        for name in objection_checks(row)
    ]


def _limit_words(name: str, state: str, detail: str, ceiling: str, row: Mapping[str, Any], said: bool = False) -> str:
    """The words of one input that holds the level (the veto named in words, no new number); `said`: the meaning
    sentence already named the objection (V ambiguous) or the missing gate verdict, so it is only referred to. A ZLK
    row's V not measured is the free-ω gate without a verdict (`policy`: gate_unavailable)."""
    if name == 'V' and state == 'not_measured' and vcc.text(row.get('zlk_gate')):
        return 'that missing gate verdict' if said else 'the free-ω gate gives no verdict'
    if name == 'V' and state == 'ambiguous':
        return 'that measured objection' if said else f'the measured objection that {_join(_objection_names(row)) or "a test fails"}'
    if name in ('veto', 'thin_veto'):
        veto = detail.strip('()') or vcc.text(row.get('veto'))
        thin = ' with a thin margin' if state == 'thin' or name == 'thin_veto' else ''
        what = VETO_BY_REASON.get((veto, vcc.text(row.get('reason'))), VETO_WORDS.get(veto, 'a measured contradiction'))
        if veto == 'owner' and vcc.text(row.get('owner')):  # what was measured, not 'the owner found'
            what = f'driven by {vcc.text(row.get("owner"))}'
        return f'the status rests on a veto, {what}{thin}, which allows at most {ceiling}'
    if name == 'table':
        if vcc.text(row.get('reason')) == 'subresonance_librates':
            return f'the status is taken from a sibling angle, and that rule fixes the level at {state}'
        return f'the rule that decided fixes the level at {state}'
    if name == 'N' and vt.finite(state):
        return f'only {vt.fmt(float(state))} {"cycle" if vt.fmt(float(state)) == "1" else "cycles"} counted'
    if (name, state) in LIMIT_STATE_WORDS:
        return LIMIT_STATE_WORDS[(name, state)]
    return vc.plain(vcc.pill_text(name, state, detail, row)).split(' (')[0]


def limit_sentence(row: Mapping[str, Any], level: str, said: bool = False) -> str:
    """The last sentence: 'Nothing limits the confidence.' at certain; 'The confidence stops at high: the separatrix
    margin is not measured and the result depends on the filter cutoff.' from the inputs whose ceiling is the shown
    level (`verdict_column_common.pills`), a veto last after a semicolon; '' without a level."""
    if not level:
        return ''
    if level == 'certain':
        return NOTHING_LIMITS
    held = [pill.key for pill in vcc.holding(vcc.pills(row, level))]
    found = _inputs(row)
    names = [name for name in dict.fromkeys(held) if name in found]
    vetoes = [name for name in names if name in ('veto', 'thin_veto')]
    words = list(dict.fromkeys(w for w in (_limit_words(n, *found[n], row, said) for n in names if n not in vetoes) if w))
    veto = list(dict.fromkeys(w for w in (_limit_words(n, *found[n], row, said) for n in vetoes) if w))
    joined = '; '.join(words) if len(words) > 1 and any(' and ' in word for word in words) else _join(words)
    body = '; '.join(part for part in (joined, *veto) if part)
    return f'The confidence stops at {level}: {body}.' if body else ''


def _would(status: str, items: Sequence[str]) -> str:
    """'Needed: …' (uncertain, invalid) or 'Would change the verdict: …' over the items; '' without one."""
    items = [item.strip().rstrip('.') for item in items if item and item.strip()]
    if not items:
        return ''
    head = NEEDED if status in ('uncertain', 'invalid') else WOULD_CHANGE
    return head + '; '.join(items) + '.'


def _last_line(status: str, no_change: str = '', change: Sequence[str] = (), checks: Sequence[str] = ()) -> str:
    """The last line from its parts: what a longer record cannot do (`no_change`), what would change the verdict and
    the next checks; an uncertain or invalid row says all of it under 'Needed: …'."""
    if status in ('uncertain', 'invalid'):
        return _would(status, [*change, *checks])
    checks = [item.strip().rstrip('.') for item in checks if item and item.strip()]
    parts = [NO_CHANGE + no_change.rstrip('.') + '.' if no_change else '', _would(status, change)]
    parts.append(NEXT_CHECK + '; '.join(checks) + '.' if checks else '')
    return ' '.join(part for part in parts if part)


# --------------------------------------------------------------------------------------------- MMR


class _Mmr:
    """The facts one MMR template reads: the `MMRFacts`, the row and the policy's inputs."""

    def __init__(self, f, row: Mapping[str, Any]):
        self.f, self.row = f, row
        self.v, self.a, self.s = f.v, f.a, f.s
        self.comm = f.v.commensurability or f.a.resonance.key
        self.key = f.a.resonance.key
        self.inputs = _inputs(row)
        self.status, self.reason = str(f.v.status), str(f.v.reason)
        self.veto, self.detail, self.angle = vcc.text(f.v.veto), vcc.text(f.v.origin_detail), str(f.v.angle_status)
        self.decided_by = vcc.text(row.get('pol_decided_by')).split(';')[0]

    def counted(self) -> List[Mapping[str, Any]]:
        """The episodes the status counts (episodes.csv `counted`), else the accepted ones the text counts."""
        rows = [row for row in self.f.episodes.rows if vt.episode_accepted(row) and true(row.get('counted'))]
        rows = rows or list(self.f.episodes.counted)
        return sorted(rows, key=lambda row: vt._float(row.get('t_start_yr')))

    def cycles(self) -> str:
        state = self.inputs.get('N', ('', '', ''))[0]
        return vt.fmt(state) if vt.finite(state) else ''

    def entry(self, name: str):
        return self.f.entry(name)


def _centre(row: Mapping[str, Any]) -> str:
    """The deciding episode's center as Key facts prints it (`centre_text`)."""
    return centre_text(row.get('ep_centre_mod_2pi_deg'))


def _record(row: Mapping[str, Any], span_yr: Any = None) -> str:
    """'the whole 100 kyr' (the record's span), 'the whole record' without one."""
    span = span_yr if span_yr is not None else row.get('smp_span_yr')
    return f'the whole {vt.years(span)}' if vt.finite(span) else 'the whole record'


def _cycles_words(cycles: str, period: Any) -> str:
    """': 323 cycles counted, period 308 yr' ('' without a count)."""
    if not cycles:
        return ''
    return f': {cycles} {"cycle" if cycles == "1" else "cycles"} counted' + (f', period {vt.years(period)}' if vt.finite(period) else '')


LIBRATION_SUBTYPE = {
    'modulated': ' with a modulated center',
    'moving_centre': ' about the moving center of its well',
}


def _regime_words(m: _Mmr) -> str:
    """' through the 1:1 regimes quasi-satellite → compound → horseshoe' from the verdict's orbit type (the header's
    'orbit: …'), else the regime change in general words."""
    orbit = vcc.text(getattr(m.v, 'orbit_type', ''))
    if m.f.coorbital and orbit.startswith('compound: '):
        return f' through the 1:1 regimes {orbit[len("compound: "):]}'
    return ' through a change of its 1:1 regime' if m.f.coorbital else ' through regimes without a circulation between them'


def _seen_libration(m: _Mmr) -> str:
    """'σ librates about 0° over the whole 100 kyr: 323 cycles counted, period 308 yr (b, k).' When the counted cycles
    do not span the whole record (the edges within a libration period), their share as Key facts prints it."""
    f, row = m.f, m.row
    subtype = str(m.v.angle_subtype)
    how = _regime_words(m) if subtype == 'regime_change' else LIBRATION_SUBTYPE.get(subtype, '')
    centre = _centre(row)
    about = f' about {centre}' if centre and subtype not in ('moving_centre', 'regime_change') else ''
    share = f.episodes.covered_fraction() if f.episodes.accepted else float('nan')
    spans = f', its cycles spanning {vt.percent(share)} of it' if vt.finite(share) and vt.percent(share) != '100%' else ''
    count = _cycles_words(m.cycles(), row.get('ep_period_med_yr'))
    if not count and not about and vt.finite(row.get('ep_n_tp')):
        count = f': {vt.fmt(row.get("ep_n_tp"))} turning points'
    portrait = M_EPISODE_PORTRAIT if f.coorbital else M_PORTRAIT
    return f'σ librates{how}{about} over {_record(row)}{spans}{count} {_panels(M_MOD, portrait)}.'


def _seen_episodic(m: _Mmr) -> str:
    counted = m.counted()
    if not counted:
        return f'σ librates in part of the record only {_panels(M_ANGLE, M_MOD)}.'
    span = vt._float(m.row.get('smp_span_yr'))
    share = _share(vt.Episodes(counted, span).covered_fraction(), span)
    end = ', still librating at the end' if true(m.row.get('pol_last_reaches_end')) else ''
    if len(counted) == 1:
        return f'σ librates only during {_span(counted[0])}{share}{end} {_panels(M_ANGLE, M_MOD)}.'
    return f'σ librates in {len(counted)} episodes, {_spans(counted)}{share}{end} {_panels(M_ANGLE, M_MOD)}.'


def _uncounted_runs(m: _Mmr) -> List[Mapping[str, Any]]:
    return [row for row in m.f.episodes.accepted if not true(row.get('counted'))]


def _seen_circulation(m: _Mmr) -> str:
    f, row = m.f, m.row
    if str(m.v.angle_subtype) == 'alias':
        rotation = row.get('smp_step_rotation_deg_max')
        return (
            f'The output step folds σ {_panels(M_MOD)}: σ turns {vt.fmt(rotation)}° per step, so the wrapped picture shows '
            f'the step, not σ; unwrapped, σ circulates {_panels(M_ANGLE)}.'
        )
    if f.rule == 'G4c':
        return f'σ circulates past the planet except for one swing in a co-orbital well {_panels(M_ANGLE, M_AXIS)}.'
    if f.rule == 'G4h':
        return f'σ circulates but for one half-swing between passages over the top {_panels(M_ANGLE, M_MOD)}.'
    runs = _uncounted_runs(m)
    if runs:
        # the bound of 'or less' is the largest swing of any run (its max half-amplitude, else its median)
        largest = [vt._float(r.get('half_amp_max_deg' if vt.finite(r.get('half_amp_max_deg')) else 'half_amp_med_deg')) for r in runs]
        amplitude = vt.fmt(max((x for x in largest if math.isfinite(x)), default=float('nan')))
        what = 'one short run' if len(runs) == 1 else f'{len(runs)} short runs'
        small = f' of small swings ({amplitude}° or less)' if vt.finite(amplitude) and float(amplitude) < 10 else ''
        return f'σ circulates {_panels(M_ANGLE, M_MOD)}, apart from {what}{small} that the status does not count.'
    made, passed = row.get('geo_n_revolutions'), row.get('geo_centre_max_pass_rev')
    turns = ''
    if vt.finite(made) and float(made) >= 1:  # the Numbers' 'revolutions of σ'
        turns = f': it makes {vt.fmt(made)} revolutions over the record'
    elif vt.finite(passed):
        turns = f': its center (the mean over a cycle) passes {vt.fmt(passed)} revolutions one way'
    return f'σ circulates {_panels(M_ANGLE, M_MOD)}{turns}.'


def _revolutions(value: Any) -> str:
    """'0.939 of a revolution', '1 revolution', '3.2 revolutions'."""
    text = vt.fmt(abs(float(value)))
    if abs(float(value)) < 1 and text != '1':
        return f'{text} of a revolution'
    return f'{text} revolution{"" if text == "1" else "s"}'


def _seen_drift(m: _Mmr) -> str:
    net = m.row.get('geo_centre_net_rev')
    moves = f' that drifts {_revolutions(net)} over the record' if vt.finite(net) else ' that drifts'
    return f'σ oscillates about a center{moves} {_panels(M_ANGLE, M_CENTRE)}.'


def _seen_near_separatrix(m: _Mmr) -> str:
    if str(m.v.angle_subtype) == 'pattern_b':
        return f'σ has no libration: in each revolution it stalls near the top of its potential {_panels(M_MOD, M_ENERGY)}.'
    return f'σ has no libration: it slips over the top of its potential {_panels(M_MOD, M_ENERGY)}.'


def _seen_insufficient(m: _Mmr) -> str:
    f, row = m.f, m.row
    rule = f.angle_rule or vt.insufficient_angle_rule(f)
    if f.rule == 'G4c':
        return f'σ makes one swing in a co-orbital well that the record’s edge cuts {_panels(M_ANGLE, M_AXIS)}.'
    if rule == 'G5w' or vt.slow_swing_decided(f):
        amplitude = row.get('geo_slow_swing_amp_deg')
        swing = f' and swings {vt.fmt(amplitude)}° in between' if vt.finite(amplitude) else ''
        return (
            f'σ never closes a libration: its slow center turns back twice{swing}, half of one slow swing ' f'{_panels(M_ANGLE, M_CENTRE)}.'
        )
    if rule == 'G5a':
        return f'σ never closes a libration: its slow center turns back once, part of one slow swing {_panels(M_ANGLE, M_CENTRE)}.'
    if str(m.v.angle_subtype) == 'stall':
        return f'σ stalls without closing a cycle {_panels(M_ANGLE, M_RATE)}.'
    n_tp = row.get('geo_n_tp')
    many = f'only {vt.fmt(n_tp)} turning points' if vt.finite(n_tp) else 'too few turning points'
    return f'σ has {many}, too few to close one cycle {_panels(M_ANGLE, M_MOD)}.'


SEEN_MMR = {
    'libration': _seen_libration,
    'episodic': _seen_episodic,
    'circulation': _seen_circulation,
    'drift': _seen_drift,
    'near_separatrix': _seen_near_separatrix,
    'insufficient_time': _seen_insufficient,
}


def _seen_slow(m: _Mmr) -> str:
    """A kinematic libration on the slow scale (rule K3s): where the slow swing is and how far it goes."""
    row = m.row
    swing = row.get('kin_meas_swing_deg')
    how = f' swings ±{vt.fmt(swing)}°' if vt.finite(swing) else ' swings'
    where = vt.interval(row.get('slow_t0_yr'), row.get('slow_t1_yr'))
    return f'The swing that counts is on the slow scale: during {where} the center of σ{how} {_panels(M_CENTRE, M_RELATED)}.'


def seen_mmr(m: _Mmr) -> str:
    """Sentence 1: what σ does, by the angle word; a single-swing leaf (G4c/G4h) reads the circulation it judged; a
    K3s row its slow scale."""
    if m.f.rule == 'K3s':
        return _seen_slow(m)
    if m.f.rule in vt.SINGLE_LEAVES and vt.single_swing_clause(m.f) and m.reason != 'baseline_needed':
        return _seen_circulation(m)
    build = SEEN_MMR.get(m.angle)
    return build(m) if build else f'σ is not defined over the record {_panels(M_ANGLE)}.'


# -- sentence 2 by reason


def _law_words(m: _Mmr) -> str:
    """L4 in words with the number the Checks print: the 1:1 law's K/K_th (1 = the planet's own mass, Evidence), else the
    weaker half's R² when it is the margin L4 read (the halves route; a harmonic or direct route keeps no number here)."""
    response = m.a.response
    law = getattr(response, 'coorbital', None) if response is not None else None
    if m.f.coorbital:
        planet = m.a.resonance.planets[0] if m.a.resonance is not None and m.a.resonance.planets else 'the planet'
        if law is not None and vt.finite(law.scale):
            return f'ȧ follows the 1:1 force law at K/K_th {vt.fmt(law.scale)}, where 1 is {planet}’s own mass'
        return COORBITAL_FORCE_SEEN
    margin = m.f.margins.get('L4') if m.f.margins else None
    halves = (getattr(response, 'force_r2_h1', None), getattr(response, 'force_r2_h2', None)) if response is not None else ()
    if margin is not None and len(halves) == 2 and all(vt.finite(x) for x in halves) and vt.finite(margin.value):
        low = min(halves)
        if math.isclose(low, float(margin.value), rel_tol=1e-9):
            return f'ȧ follows its force law with R² {vt.fmt(low)} in the weaker half'
    return FORCE_SEEN['L4'][0]


def _energy_words(m: _Mmr) -> str:
    """R0 in words: inside one episode (W34 `zone.episode_r0`, 79653: not the deciding one), or over its windows with
    the count the Checks print ('322 of 322 windows', `verdict_text.window_count_text`)."""
    episode = getattr(m.a.energy, 'r0_episode_id', None)
    if episode is not None:
        found = next((r for r in m.f.episodes.accepted if vt._same_id(r.get('episode_id'), episode)), None)
        where = f' ({_span(found)})' if found is not None else ''
        decisive = m.f.episodes.decisive
        other = '' if decisive is None or vt._same_id(decisive.get('episode_id'), episode) else ', not the deciding one'
        return f'inside episode #{episode}{where}{other}, the cycle energy fits the potential built from a'
    counts = re.match(r'(\d+) of (\d+) eligible windows', vt.window_count_text(m.f))
    windows = f' in {counts.group(1)} of {counts.group(2)} windows' if counts else ''
    clear = ' and stays clear of its separatrix' if m.inputs.get('M', ('',))[0] == 'full' else ''
    return f'{FORCE_SEEN["R0"][0]}{windows}{clear}'


def _pendulum_sentence(m: _Mmr, head: str = '') -> str:
    """The force law (L4) and the energy (R0) that passed, each with its number: 'The semi-major axis swings in step
    with σ as the pendulum of 2J-1 predicts: ȧ follows its force law with R² 0.814 in the weaker half (e), and the
    cycle energy fits the potential built from a in 322 of 322 windows and stays clear of its separatrix (h).'; ''
    when neither passed."""
    shown = vcc.force_shown(m.row)
    parts = []
    if 'L4' in shown:
        parts.append(f'{_law_words(m)} {_panels(M_RESPONSE)}')
    if 'R0' in shown:
        parts.append(f'{_energy_words(m)} {_panels(M_ENERGY)}')
    if not parts:
        return ''
    if not head:
        head = (
            f'The semi-major axis swings in step with σ as the pendulum of {m.comm} predicts'
            if 'L4' in shown
            else f'The libration is {m.comm}’s own'
        )
    return f'{head}: {", and ".join(parts)}.'


def _strength_sentence(m: _Mmr) -> str:
    """F in words. The averaged strength (two-body): the small-swing period T₀ its potential predicts against T₀ from
    the observed cycles and the strength ratio in its band, as the Evidence rows print them (never 'within a factor of
    two': the test is on μ needed/μ theory = (T₀ theory/T₀)²). A three-body e^q strength is an order-of-magnitude scale:
    only that it agrees, marked OPTIONAL (dropped first). '' when F did not pass."""
    if 'F' not in vcc.force_shown(m.row) or m.a.strength is None:
        return ''
    p, s = vt._Printer(m.row), m.s
    if str(m.a.strength.method) in DIRECT_STRENGTH:
        if not p.has('str_P_lib_theory_min_kyr', 'str_P_lib_theory_max_kyr', 'str_T_obs_yr', 'str_ratio_all_min', 'str_ratio_all_max'):
            return f'The strength of its averaged term can make cycles of this period: {m.comm} is strong enough.'
        values = [abs(p.value('str_P_lib_theory_min_kyr')) * vt.KYR, abs(p.value('str_T_obs_yr'))]
        scale, unit = (1 / vt.KYR, 'kyr') if min(values) >= vt.KYR else (1.0, 'yr')  # one unit, as Evidence prints them
        theory = p.span('str_P_lib_theory_min_kyr', 'str_P_lib_theory_max_kyr', vt.KYR * scale, unit=f' {unit}')
        observed = f'{p.num("str_T_obs_yr", scale)} {unit}'
        low, high, limit = p.value('str_ratio_all_min'), p.value('str_ratio_all_max'), s.mmr_strength_check_ratio
        where = 'inside' if low >= 1 / limit and high <= limit else 'overlapping'
        ratio = p.span('str_ratio_all_min', 'str_ratio_all_max', form=vt.sci)
        return (
            f'Its averaged potential predicts a small-swing period of {theory}, against {observed} derived from the '
            f'observed cycles, so these cycles need {ratio} times its theoretical strength, {where} the accepted band '
            f'{vt.fmt(1 / limit)}–{vt.fmt(limit)}: {m.comm} is strong enough to make them.'
        )
    ratio = f'; these cycles need {vt.sci(m.row.get("str_ratio"))} times it' if vt.finite(m.row.get('str_ratio')) else ''
    return f'{OPTIONAL}The strength of {m.comm} is known only as an order-of-magnitude scale{ratio}, which agrees but is no measurement.'


def _t_dynamic(m: _Mmr) -> List[str]:
    pendulum, strength = _pendulum_sentence(m), _strength_sentence(m)
    if not pendulum and not strength:
        return [f'Nothing measured {m.comm}’s own force, so the status rests on the shape of σ.']
    if not pendulum and strength.startswith(OPTIONAL):  # the scale alone shows nothing (2348, 24820)
        return _scale_only(m)
    out = [pendulum, strength]
    if m.status == 'transient' and m.angle != 'episodic':  # sentence 1 says 'only during …' for an episodic σ
        out.append('It is transient, not resonant, because σ does not librate over the whole record.')
    return [sentence for sentence in out if sentence]


def _scale_only(m: _Mmr) -> List[str]:
    """Only the order-of-magnitude strength scale agrees: no physical measurement, the shape decides (2348, 24820)."""
    ratio = f' (these cycles need {vt.sci(m.row.get("str_ratio"))} times it)' if vt.finite(m.row.get('str_ratio')) else ''
    return [
        f'Neither the force law nor the energy test could be read; only the order-of-magnitude strength scale of {m.comm} '
        f'agrees{ratio}, and a scale is no measurement, so the status rests on the shape of σ.'
    ]


# a measured objection the level reads (`policy.OBJECTION_*`) in the Why's words, with its panel: MMR panels for the MMR
# objections, secular/ZLK panels for the free-ω gate's (`_t_long_geometric`; the e-i integral's own sentence is
# `_ei_sentence`)
OBJECTION_SEEN = {
    pol.OBJECTION_FORCE_LAW: ('ȧ does not follow its force law', M_RESPONSE),
    pol.OBJECTION_POTENTIAL: ('the energy of the cycles does not fit the potential built from a', M_ENERGY),
    pol.OBJECTION_IDENTITY: ('σ̇ does not follow a', M_RATE),
    pol.OBJECTION_STRENGTH: ('its strength does not match the period of the cycles', ''),
    pol.OBJECTION_ONSET: ('a already swung with comparable strength before the episode', M_AXIS),
    pol.OBJECTION_STRENGTH_SCALE: ('the order-of-magnitude strength scale is far from what the period needs', ''),
    pol.OBJECTION_FORCE_LINE: ('ȧ follows a forced line at the period, not the libration', M_RESPONSE),
    pol.OBJECTION_OWNER: ('another resonance may drive a', M_RELATED),
    pol.OBJECTION_KINEMATIC: ('the perihelion term may bound σ', M_RELATED),
    pol.OBJECTION_ZLK_GATE: ('the free ω winds', L_DRIFT),
    pol.OBJECTION_ZLK_FREE_E: ('the free e stays below the forced e over the record', L_VECTOR),
}
# the objections of the free-ω gate itself (a kinematic gate that pins no episode, the free e below the forced e with an
# undefined phase); the e-i integral is no gate measurement
GATE_OBJECTIONS = (pol.OBJECTION_ZLK_GATE, pol.OBJECTION_ZLK_FREE_E)


def _objections(m: _Mmr) -> Tuple[str, Tuple[str, ...]]:
    names = objection_checks(m.row)
    words = [OBJECTION_SEEN.get(name, (vcc.OBJECTION_WORDS.get(name, vt.snake_words(name)), ''))[0] for name in names]
    letters = tuple(dict.fromkeys(OBJECTION_SEEN.get(name, ('', ''))[1] for name in names if OBJECTION_SEEN.get(name, ('', ''))[1]))
    return _join(words), letters


def _t_geometric(m: _Mmr) -> List[str]:
    shown = vcc.force_shown(m.row)
    objections, objection_letters = _objections(m)
    if m.decided_by == pol.DECIDED_OBJECTION or objections:
        passed = []
        if 'L4' in shown:
            passed.append(_law_words(m))
        if 'R0' in shown:
            passed.append(_energy_words(m))
        but = f'{_join(passed)}, but ' if passed else ''
        letters = ((M_RESPONSE,) if 'L4' in shown else ()) + ((M_ENERGY,) if 'R0' in shown else ())
        return [
            f'The libration is not shown to be {m.comm}’s own: {but}{objections or "a measured test objects"} '
            f'{_panels(*dict.fromkeys(letters + objection_letters))}, so the status follows the shape of σ.'.replace(' ,', ',')
        ]
    if m.decided_by == pol.DECIDED_SCALE:
        return [
            f'The strength of {m.comm} is known only as an order-of-magnitude scale, and it is far from what the period '
            'needs; a scale is no measurement, so the status follows the shape of σ.'
        ]
    pendulum, strength = _pendulum_sentence(m), _strength_sentence(m)
    if not pendulum and strength.startswith(OPTIONAL):
        return _scale_only(m)
    if pendulum or strength:
        return [s for s in (pendulum, strength.lstrip(OPTIONAL)) if s]
    return [
        f'Nothing measured {m.comm}’s own force: neither the force law nor the energy could be read, so the status rests on the shape of σ.'
    ]


def _t_separatrix(m: _Mmr) -> List[str]:
    if vt.has_mark(m.v, 'near_separatrix_without_zone'):
        out = [
            f'That is the pattern of motion near the separatrix of {m.comm}; the separatrix zone itself is only estimated, '
            'so it neither confirms nor excludes this.'
        ]
    else:
        out = [f'Its cycles lie in the separatrix zone of {m.comm} {_panels(M_ENERGY)}: near the edge of the well, not inside it.']
    pendulum = _pendulum_sentence(m, head=f'The force is {m.comm}’s own')
    return out + ([pendulum] if pendulum else [])


def _t_subresonance(m: _Mmr) -> List[str]:
    sub = m.a.subresonances
    accepted = sub.accepted if sub is not None else None
    key = accepted.spec.key if accepted is not None else (m.v.resonant_argument or m.v.owner)
    return [
        f'Its sibling {key}, an angle of the same commensurability with other eccentricity and inclination terms, '
        f'librates about a fixed center while the center of σ moves {_panels(M_RELATED)}; siblings of one '
        'commensurability are one resonance, so σ takes the sibling’s status.'
    ]


def psi_formula(definition: Any) -> str:
    """'Ψ = 17σ − 3φ' from the column own_psi_def ('17*sigma-3*phi'); '' without one."""
    match = vt.PSI_DEF.match(vcc.text(definition))
    if match is None:
        return ''
    u, sign, k = int(match.group(1)), match.group(2), int(match.group(3))
    if sign == '-':
        k = -k
    first = {1: 'σ', -1: '−σ'}.get(u, f'{u}σ'.replace('-', '−'))
    second = '' if not k else f' {"+" if k > 0 else "−"} {"" if abs(k) == 1 else abs(k)}φ'
    return f'Ψ = {first}{second}'


def _t_induced(m: _Mmr) -> List[str]:
    f, owner = m.f, m.a.owner
    key = m.v.owner or m.v.cause_term
    r2 = owner.accepted.r2_median if owner is not None and owner.accepted is not None else None
    fit = f' with R² {vt.fmt(r2, 2)}' if vt.finite(r2) else ''
    role = vcc.text(m.v.cause_role)
    if vt.owner_confirmed(f) and role == 'driver':
        first = (
            f'The libration is borrowed from {key}, which circulates and drives a forced wave on a: its force law fits ȧ'
            f'{fit} {_panels(M_RESPONSE, M_RELATED)}.'
        )
    elif vt.owner_confirmed(f):
        first = (
            f'The libration is borrowed from {key}, which librates over σ’s episode and drives a: its force law fits ȧ'
            f'{fit} {_panels(M_RESPONSE, M_RELATED)}.'
        )
    else:
        first = (
            f'The libration is borrowed from {key}, a candidate owner that fits a{fit} but is not shown to cover σ’s '
            f'episode {_panels(M_RELATED)}.'
        )
    identity = vt.identity_line(f).split(':')[0]
    psi = psi_formula(m.row.get('own_psi_def'))
    named = f', with φ the angle of {key}' + (f' and {psi}' if psi and 'Ψ' in identity else '')
    through = f'{identity}{named}' if identity else 'the identity of the two angles'
    return [first, f'σ follows it through {through}, so the libration is not {m.comm}’s own.']


def _psi_name(m: _Mmr) -> str:
    match = vt.PSI_DEF.match(vcc.text(m.row.get('own_psi_def')))
    if match is None or not int(match.group(1)):
        return ''
    u = int(match.group(1))
    return f'Ψ/{u}' if u > 0 else f'Ψ/({u})'


def _b1_words(m: _Mmr) -> str:
    entry = m.entry(vt.B1_AMPLITUDE)
    if entry is None or entry.result is not False or not vt.finite(entry.value):
        return ''
    return f'it carries only {vt.sci(entry.value)} of the swing a resonant libration of this size would need'


def _t_kinematic(m: _Mmr) -> List[str]:
    f, row = m.f, m.row
    if m.detail == 'forced_wave_background':
        return [
            'They are the background’s forced wave, not librations: throughout the record σ carries a small steady wave '
            f'at their period, and these runs are where it shows {_panels(M_ANGLE, M_RELATED)}; {m.comm} does not make them.'
        ]
    p = int(m.a.resonance.body_varpi_coefficient)
    term = vt.varpi_term(p, '*' if m.a.resonance.retrograde else '') if p else 'perihelion'
    period, reference, name, _ = vt.secular_reference(f)
    if f.rule == 'K3s' or vt.slow_kinematic(row, m.s.mmr_kin_period_tol):
        cycle = f', which turns with {name} ({vt.years(reference)})' if name and vt.finite(reference) else ''
        return [
            f'Kinematic (forced) libration: the slow swing is bounded by σ’s {term} term{cycle}, not by {m.comm}’s force '
            f'{_panels(M_CENTRE, M_RELATED)}.'
        ]
    # the secular period matched σ's (origin_detail secular_cycle) and the predicted pϖ half-swing against σ's: both
    # columns the Numbers print (KINEMATICS, LIBRATION)
    details = []
    if name and vt.finite(reference):
        same = vt.years(period) == vt.years(reference)
        of_sigma = ', the period of σ' if same else (f' (σ’s period {vt.years(period)})' if vt.finite(period) else '')
        details.append(f'swings with {name} = {vt.years(reference)}{of_sigma}')
    if vt.finite(row.get('kin_pred_swing_deg')) and vt.finite(row.get('ep_half_amp_med_deg')):
        details.append(f'predicts a half-swing of {vt.fmt(row["kin_pred_swing_deg"])}° against σ’s {vt.fmt(row["ep_half_amp_med_deg"])}°')
    that = f'; that part {", and ".join(details)}' if details else ''
    out = [f'Kinematic (forced) libration: the swing comes from σ’s {term} part, not from {m.comm}’s force{that} ' f'{_panels(M_RELATED)}.']
    b1 = _b1_words(m)
    if b1:
        out.append(f'The semi-major axis takes no part: {b1} {_panels(M_AXIS, M_RESPONSE)}.')
    return out


def _t_kinematic_drift(m: _Mmr) -> List[str]:
    p = int(m.a.resonance.body_varpi_coefficient)
    term = vt.varpi_term(p) if p else 'perihelion'
    corr = m.entry(vt.K2_CORR)
    follows = f' (correlation {vt.fmt(corr.value, 2)})' if corr is not None and vt.finite(corr.value) else ''
    out = [
        f'Kinematic drift: σ’s slow center follows its {term} term{follows}, not {m.comm} {_panels(M_CENTRE, M_RELATED)}; '
        'the short swings ride on that drift.'
    ]
    b1 = _b1_words(m)
    if b1:
        out.append(f'The semi-major axis takes no part: {b1} {_panels(M_AXIS, M_RESPONSE)}.')
    return out


def _path_entry(m: _Mmr, suffix: str):
    """The decision-path entry of rule 5s whose name ends with `suffix` (it is recorded before step 5 or in the
    branch, as `verdict_text._strength_entry_object` reads it), with a value; None without one."""
    return next((e for e in m.v.decision_path if e.name.startswith('5s') and e.name.endswith(suffix) and vt.finite(e.value)), None)


def _t_strength(m: _Mmr) -> List[str]:
    direct = _path_entry(m, vt.STRENGTH_DIRECT)
    entry = direct or _path_entry(m, vt.STRENGTH_THREE_BODY)
    times = f' {vt.fmt(entry.value)} times' if entry is not None else ''
    what = (
        f'the strength of {m.comm}’s averaged resonant term' if direct is not None or entry is None else f'the strength scale of {m.comm}'
    )
    return [
        f'Cycles of the period these swings show would need{times} {what}: the term is far too weak to make them, so '
        'these swings are not its libration.'
    ]


A_SCALE = 'an a-scale is half the swing of a'  # the Numbers' gloss of smp_far_ratio: |⟨a⟩ − a*| over a's half-swing


def _t_alias(m: _Mmr) -> List[str]:
    far = m.entry(vt.C5_FAR)
    measured = far is not None and vt.finite(far.value)
    distance = f'{vt.fmt(far.value)} a-scales' if measured else 'far'
    if m.status == 'invalid':
        why = []
        branch = m.entry(vt.C5_BRANCH)
        if branch is None or not branch.result:
            why.append('several alias branches of the step fit the range of a')
        if far is None or far.result is not True:
            why.append(
                f'⟨a⟩ lies only {distance} from the resonance' if measured else 'the distance of ⟨a⟩ from the resonance is not measured'
            )
        although = f', although ⟨a⟩ lies {distance} from a* ({A_SCALE})' if measured and far.result is True else ''
        return [f'A libration hidden by the step cannot be excluded: {_join(why) or "the alias test does not pass"}{although}.']
    gloss = f', {A_SCALE.replace(" is ", " being ")}' if measured else ''
    near = 'a never comes near a*, so ' if measured and far.result is True and float(far.value) > 1 else ''
    return [
        f'Only one alias branch fits the range of a, and ⟨a⟩ lies {distance} from the resonance {_panels(M_AXIS)}{gloss}: '
        f'{near}the folded picture is no libration.'
    ]


def _t_circulation(m: _Mmr) -> List[str]:
    f = m.f
    if f.rule in vt.SINGLE_LEAVES:
        clause = vt.single_swing_clause(f)
        out = [_sentence(clause)] if clause else []
        if m.inputs.get('candidate', ('',))[0] == 'drawn':
            out.append('One swing closes no cycle: it is drawn as a candidate, not counted, and the status is that of the circulation.')
        return out
    if vt._forced_wave_word(f):
        return [f'Its short loops are {vt.forced_wave_source(f)}; a body whose σ circulates is not in {m.comm}.']
    return [f'A body whose σ circulates is not in {m.comm}; no other test is needed.']


def _t_no_potential(m: _Mmr) -> List[str]:
    return [
        f'σ turns back at times, but the energy of its swings does not fit a potential built from a {_panels(M_ENERGY)}: no well of {m.comm} holds it.'
    ]


def _t_modulated(m: _Mmr) -> List[str]:
    return [f'σ’s center passes at least one revolution {_panels(M_CENTRE)}: there is no center to librate about, so σ is not in {m.comm}.']


def _t_drift_unknown(m: _Mmr) -> List[str]:
    tested = []
    if str(m.v.checks.get('check_mmr_kinematic', '')) == 'pass':
        tested.append('the perihelion term does not carry the drift')
    if m.a.owner is not None and str(m.a.owner.check) == 'pass':
        tested.append('no other resonance drives a')
    out = ['No tested mechanism explains the drift of the center.']
    if tested:
        out.append(f'Measured: {_join(tested)}.')
    return out


def _t_baseline(m: _Mmr) -> List[str]:
    f, row = m.f, m.row
    qs = vt._held(f, QS_STALL_ENTRY)
    if qs:
        return [
            'The filter cutoff hides the libration: a longer record would not decide, the output step must hold the quasi-satellite period.'
        ]
    cause = vcc.text(row.get('base_cause'))
    if cause:
        return [
            f'The record already holds the time the period needs; what is missing: {BASE_CAUSE_WORDS.get(cause, vt.snake_words(cause))}.'
        ]
    if f.rule == 'G4c':
        clause = vt.single_swing_clause(f)
        return [_sentence(clause)] if clause else []
    rule = f.angle_rule or vt.insufficient_angle_rule(f)
    if rule == 'G5w' or vt.slow_swing_decided(f):
        ratio = row.get('geo_slow_swing_period_ratio')
        period = (
            f'its period is {vt.fmt(ratio)} × the theoretical period of the {m.comm} pendulum at this amplitude'
            if vt.finite(ratio)
            else f'its period matches the pendulum of {m.comm}'
        )
        return [
            f'The swing behaves like the well’s own: σ̇ follows a, and {period} {_panels(M_RATE, M_AXIS)}.',
            'Half of one swing still cannot tell a slow libration from a slow circulation through the resonance.',
        ]
    if rule == 'G5a':
        return ['Part of one slow swing cannot tell a drift from a slow libration.']
    return ['Too few turning points close no cycle: the record cannot tell a libration from a slow circulation.']


def _t_one_leg(m: _Mmr) -> List[str]:
    clause = vt.single_swing_clause(m.f)
    return [_sentence(clause)] if clause else ['One leg near the planet neither shows a capture nor excludes one.']


def _t_origin_unknown(m: _Mmr) -> List[str]:
    return [f'σ librates, but no measurement shows that {m.comm} makes it.']


def _t_unusable(m: _Mmr) -> List[str]:
    return [f'The series cannot be read: {vt.snake_words(m.reason)}; no angle is judged.']


MMR_TEMPLATES: Dict[str, Callable[[_Mmr], List[str]]] = {
    Reason.DYNAMIC_LIBRATION: _t_dynamic,
    Reason.DYNAMIC_EPISODE: _t_dynamic,
    Reason.GEOMETRIC_LIBRATION: _t_geometric,
    Reason.GEOMETRIC_EPISODE: _t_geometric,
    Reason.SEPARATRIX_ZONE: _t_separatrix,
    Reason.CIRCULATION: _t_circulation,
    Reason.ALIAS: _t_alias,
    Reason.NO_RESONANT_POTENTIAL: _t_no_potential,
    Reason.MODULATED_CIRCULATION: _t_modulated,
    Reason.KINEMATIC_LIBRATION: _t_kinematic,
    Reason.KINEMATIC_DRIFT: _t_kinematic_drift,
    Reason.INDUCED_LIBRATION: _t_induced,
    Reason.SUBRESONANCE_LIBRATES: _t_subresonance,
    Reason.BASELINE_NEEDED: _t_baseline,
    Reason.DRIFT_MECHANISM_UNKNOWN: _t_drift_unknown,
    Reason.ORIGIN_UNKNOWN: _t_origin_unknown,
    Reason.IMPLAUSIBLE_STRENGTH: _t_strength,
    Reason.ONE_COORBITAL_LEG: _t_one_leg,
    Reason.ONE_HALF_LIBRATION: _t_one_leg,
    Reason.UNUSABLE_SERIES: _t_unusable,
    Reason.UNPHYSICAL: _t_unusable,
    Reason.UNDEFINED_PHASE: _t_unusable,
    Reason.UNDERSAMPLED: lambda m: ['The output step is too coarse for the filter: no cutoff keeps the shortest libration.'],
    Reason.INVALID_INPUT: lambda m: ['The angle is no resonance of this body: it was refused before the analysis.'],
    Reason.GATE_UNAVAILABLE: _t_origin_unknown,
}

# reasons whose angle is bounded and whose Why names what was ruled out (sentence 3)
EXCLUDED_REASONS = (
    'dynamic_libration',
    'dynamic_episode',
    'geometric_libration',
    'geometric_episode',
    'separatrix_zone',
    'origin_unknown',
    'kinematic_libration',
    'kinematic_drift',
)


def _thin(m: _Mmr, margin_id: str) -> Any:
    """The margin `margin_id` when it passed only thinly (its value), else None."""
    margin = m.f.margins.get(margin_id) if m.f.margins else None
    return margin.value if margin is not None and str(margin.margin) == 'thin' and vt.finite(margin.value) else None


def _sibling_words(m: _Mmr) -> Tuple[str, str]:
    """(ruled out, not fully ruled out) of the sibling scan that passed: 'no sibling angle of the same
    commensurability librates about a steadier center (29 scanned)'; a thin S margin is the Checks' failing reading
    with its range ratio."""
    ratio = _thin(m, 'S')
    if ratio is not None:
        return '', f'a sibling angle’s center is as still as σ’s or stiller (range ratio {vt.fmt(ratio)})'
    n = m.row.get('sub_n_scanned')
    scanned = f' ({vt.fmt(n)} scanned)' if vt.finite(n) else ''
    return f'no sibling angle of the same commensurability librates about a steadier center{scanned}', ''


def _owner_words(m: _Mmr) -> Tuple[str, str]:
    """(ruled out, not fully ruled out) of the owner search that passed: 'none of 38 neighboring resonances drives a
    (the best, 5S-1+0-4, fits a with R² 0.06)'; a thin W margin: the best one's R² between the no-owner level and an
    owner's bar, as the Checks print them."""
    owner, s = m.a.owner, m.s
    best = vt._best_candidate(owner.candidates)
    r2 = vt.fmt(best.r2_median, 2) if best is not None and vt.finite(best.r2_median) else ''
    if _thin(m, 'W') is not None and best is not None and r2:
        return '', (
            f'{best.spec.key} fits a with R² {r2}, below the {vt.fmt(s.mmr_owner_r2_min)} an owner needs but above the '
            f'{vt.fmt(s.mmr_w_r2_none)} that rules one out'
        )
    n = int(owner.n_candidates or 0)
    fit = f' (the best, {best.spec.key}, fits a with R² {r2})' if best is not None and r2 else ''
    if n == 1 and best is not None:
        return f'the one neighboring resonance tested, {best.spec.key}, does not drive a' + (f' (R² {r2})' if r2 else ''), ''
    return (f'none of {n} neighboring resonances drives a{fit}' if n else 'no other resonance lies near enough to drive a'), ''


def excluded_mmr(m: _Mmr) -> str:
    """Sentence 3, positive clauses with their numbers: what the measurements ruled out (the sibling scan, the owner
    search, a carries the swing), what they did not fully rule out (a thin S or W margin, as the Checks and the pill
    'alternatives partly ruled out' say), and whether another resonance drives a when that search did not run (with
    its reason; never on a kinematic row, where a does not move); '' for other reasons. No perihelion clause for an
    angle without a ϖ term (1:1)."""
    if m.reason not in EXCLUDED_REASONS or m.detail == 'forced_wave_background':
        return ''
    f, a = m.f, m.a
    kinematic = m.reason.startswith('kinematic')
    out, partly, not_tested = [], [], ''
    if a.subresonances is not None and str(a.subresonances.check) == 'pass':
        clean, thin = _sibling_words(m)
        out.append(clean)
        partly.append(thin)
    owner = a.owner
    coorbital = f.coorbital or (owner is not None and 'owner_coorbital' in owner.flags)
    if owner is not None and str(owner.check) == 'pass':
        clean, thin = _owner_words(m)
        out.append(clean)
        partly.append(thin)
    elif owner is not None and str(m.v.checks.get('check_mmr_owner', '')) == 'pass':
        # the search found a resonance that drives a, but not in σ's episode (F7 cover, P2 turns): the verdict's pass
        out.append('no other resonance drives a during σ’s episode')
    elif not coorbital and not f.derived and not kinematic:
        reason = vt.why_not_measured(f, 'W')[1]
        reason = '' if reason in ('not run', 'not measured') else f' ({reason})'
        not_tested = f'whether another resonance drives a was not tested{reason}'
    b1 = m.entry(vt.B1_AMPLITUDE)
    if not kinematic and str(m.v.checks.get('check_mmr_a_amplitude', '')) == 'pass' and b1 is not None and vt.finite(b1.value):
        times = '' if vt.fmt(b1.value) == '1' else f'{vt.fmt(b1.value)} times '
        carries = f'the semi-major axis swings {times}as far as a libration of this size needs'
        if int(a.resonance.body_varpi_coefficient):
            carries += ', so it is not a kinematic libration made by the motion of the perihelion (there a would hardly swing)'
        out.append(carries)
    parts = list(out)
    if any(partly):
        parts.append(f'not fully ruled out: {_join(partly)}')
    parts.append(not_tested)
    return '; '.join(part for part in parts if part) + '.' if any(parts) else ''


# -- would change (MMR)


def _gap_words(m: _Mmr) -> str:
    """Where σ does not librate inside the record, so that a longer record does not make it resonant: 'during 32.5–61.3 kyr'
    (the longest gap between counted episodes), 'after 537 yr' (the last one ends more than the coverage allows before
    the record's end), 'before 4.81 kyr' (the first one misses the start so, and the record grows forward only); ''
    when none holds (an extension could then change the status)."""
    counted = m.counted()
    if not counted:
        return ''
    gap = longest_gap(counted)
    if gap:
        return gap
    limit, last, first = m.s.libration_coverage_periods, counted[-1], counted[0]
    end_gap, start_gap = last.get('coverage_end_gap_P'), first.get('coverage_start_gap_P')
    if vt.finite(end_gap) and float(end_gap) > limit:
        return f'after {vt.years(last.get("t_end_yr"))}'
    if vt.finite(start_gap) and float(start_gap) > limit and vcc.text(m.v.extend_direction) in ('', 'forward'):
        return f'before {vt.years(first.get("t_start_yr"))}'
    return ''


def _clause_words(m: _Mmr) -> str:
    """The policy's extend clause with its length, in words; '' without a clause or a length."""
    clause, row = vcc.text(m.row.get('pol_extend_clause')), m.row
    to = m.v.extend_to_years if str(m.v.action) == 'extend' else None
    length = f'a record of {vt.years(to)}' if vt.finite(to) else 'a longer record'
    if clause == 'A1':
        return f'{length} that shows how the 1:1 leg cut by the record’s end closes'
    if clause == 'A2':
        return f'{length} that shows how the half-swing cut by the record’s end closes'
    if clause == 'A' and m.status == 'uncertain':
        return _baseline_need(m, length)
    if clause == 'A':
        n_last = row.get('pol_n_last')
        cycles = f' after {vt.fmt(n_last)} cycles of its last episode' if vt.finite(n_last) else ''
        return f'{length}, which shows whether σ, still librating at the end{cycles}, keeps librating'
    if clause == 'B':
        period = row.get('pol_period_dec_yr')
        return f'{length}, since the deciding period ({vt.years(period)}) is too long for this record' if vt.finite(period) else length
    if clause == 'C':
        where = ' in the last episode' if len(m.counted()) > 1 else ''
        return f'{length}, long enough to test the force law{where}, which could confirm its own force or object to it'
    return ''


def _baseline_need(m: _Mmr, length: str) -> str:
    """'3 slow periods of 76.3 kyr, about 230 kyr in all; next check: a record of 300 kyr' (`decide.baseline`)."""
    f, row = m.f, m.row
    required, period = row.get('base_required_kyr'), row.get('base_period_yr')
    if not vt.finite(required):
        return length
    rule = f.angle_rule or vt.insufficient_angle_rule(f)
    slow = rule == 'G5w' or vt.slow_swing_decided(f)
    cycles = m.s.mmr_extend_slow_cycles if slow else m.s.libration_min_turning_points / 2
    what = 'slow periods' if slow else 'periods'
    of = f' of {vt.years(period)}' if vt.finite(period) else ''
    need = f'{vt.fmt(cycles)} {what}{of}, about {vt.years(float(required) * vt.KYR)} in all'
    return f'{need}; next check: {length}' if length != 'a longer record' else need


def would_change_mmr(m: _Mmr) -> str:
    """The last line from the computed quantities only (module docstring): a circulation between the episodes that a
    longer record keeps ('A longer record cannot make it resonant: …'), what would change the verdict (both resonances
    at once), the next check (the extend clause), or all of it under 'Needed: …'."""
    gap = _gap_words(m) if m.reason in ('dynamic_episode', 'geometric_episode') else ''
    change, checks = [], []
    clause = _clause_words(m)
    if clause:
        checks.append(clause)
    elif m.status == 'uncertain' and m.reason == 'baseline_needed' and not m.f.rule == 'G4c':
        checks.append(_baseline_need(m, 'a longer record'))
    if m.inputs.get('second_capture', ('',))[0] == 'unresolved':
        psi = _psi_name(m) or 'the second angle of the identity'
        change.append(
            f'if {psi} were shown to stay bounded as well, the body would sit in both resonances at once, and the '
            f'libration would also count as {m.comm}’s own'
        )
    if m.status == 'invalid' and vt.finite(m.v.resample_step_yr):
        checks.append(f'an output step of {vt.fmt(m.v.resample_step_yr)} yr, which unfolds σ')
    return _last_line(m.status, f'σ does not librate {gap}' if gap else '', change, checks)


def mmr_why(f, row: Mapping[str, Any]) -> vc.Why:
    """The Why block of an MMR row: `f` = `verdict_text.MMRFacts`, `row` = summary ⋈ numbers."""
    m = _Mmr(f, row)
    template = MMR_TEMPLATES.get(m.reason, _t_origin_unknown)
    meaning = template(m)
    # the objection already named by the meaning sentence (`_t_geometric`) is only referred to in the limit
    said = any('objection' in sentence or 'not shown to be' in sentence for sentence in meaning)
    limit = limit_sentence({**row, 'reason': m.reason}, vcc.text(m.v.confidence), said)
    sentences = [seen_mmr(m), *meaning, excluded_mmr(m), limit]
    return vc.Why(_fit(sentences), would_change_mmr(m))


# --------------------------------------------------------------------------------------------- secular and ZLK


class _Long:
    def __init__(self, analysis, settings, row: Mapping[str, Any], episodes: Sequence[Mapping[str, Any]]):
        self.a, self.s, self.row = analysis, settings, row
        self.v = analysis.verdict
        self.zlk = str(self.v.resonance_type) == 'lidov_kozai'
        self.noun = 'ω' if self.zlk else 'The proper angle σ'  # the first mention
        self.short = 'ω' if self.zlk else 'σ'
        self.status, self.reason = str(self.v.status), str(self.v.reason)
        self.veto, self.angle = vcc.text(self.v.veto), str(self.v.angle_status)
        self.gate = getattr(analysis, 'gate', None)
        self.outcome = vcc.text(self.gate.outcome) if self.gate is not None else ''
        self.span = vt.record_span_yr(analysis)
        self.librating = sorted(vt.librating_episode_rows(analysis, episodes), key=lambda r: vt._float(r.get('t_start_yr')))
        self.inputs = _inputs(row)
        self.said = False  # the meaning named the objection or the missing gate verdict (the limit only refers to it)

    def free_turns(self) -> str:
        """The free ω's turns at its mean rate, |g − s|·T, as `verdict_text.zlk_physics_line` prints them ('' when not
        measured); not the net turns of the free elements (free_omega_revolutions)."""
        return mean_rate_turns(self.a, self.row)

    def episode_ei(self):
        """The e-i exchange of the deciding librating episode (`zlk.episode_ei`), None when not measured."""
        ei = getattr(self.gate, 'episode_ei', None) if self.gate is not None else None
        return ei if ei is not None and vt.finite(getattr(ei, 'eta_h', None)) else None

    def zlk_ei(self) -> bool:
        """Whether e and i trade as a ZLK cycle in the deciding episode (its coupling is zlk)."""
        ei = self.episode_ei()
        return ei is not None and vcc.text(ei.coupling) == 'zlk'


def mean_rate_turns(analysis, row: Mapping[str, Any]) -> str:
    """The free ω's turns over the record at its mean rate, |g − s|·T (`verdict_text.zlk_physics_line`'s number, as it
    prints it; '' when not measured). The Numbers block of a ZLK row prints it under its own label."""
    line, _ = vt.zlk_physics_line(analysis, row)
    found = re.search(r'turns (\S+) times', line)
    return found.group(1) if found else ''


def _seen_long_libration(m: _Long) -> str:
    """'ω librates about 270° over the whole 500 kyr: 12.2 cycles counted, period 39.5 kyr, half-amplitude 21.1° (a, b).'
    (the deciding episode's period and half-amplitude, the Numbers' LIBRATION values)."""
    row = m.row
    centre, cycles = _centre(row), m.inputs.get('N', ('', '', ''))[0]
    count = _cycles_words(vt.fmt(float(cycles)) if vt.finite(cycles) else '', row.get('ep_period_med_yr'))
    if count and vt.finite(row.get('ep_half_amp_med_deg')):
        count += f', half-amplitude {vt.fmt(row["ep_half_amp_med_deg"])}°'
    if not count and not centre and vt.finite(row.get('ep_n_tp')):
        count = f': {vt.fmt(row.get("ep_n_tp"))} turning points'
    about = f' about {centre}' if centre else ''
    return f'{m.noun} librates{about} over {_record(row, m.span)}{count} {_panels(L_ANGLE, L_MOD)}.'


def _seen_long_episodic(m: _Long) -> str:
    rows = m.librating
    if not rows:
        return f'{m.noun} librates in part of the record only {_panels(L_ANGLE, L_MOD)}.'
    share = _share(vt.Episodes(rows, m.span).covered_fraction(), m.span)
    if len(rows) == 1:
        return f'{m.noun} librates only during {_span(rows[0])}{share} {_panels(L_ANGLE, L_MOD)}.'
    return f'{m.noun} librates in {len(rows)} episodes, {_spans(rows)}{share} {_panels(L_ANGLE, L_MOD)}.'


def _seen_long_insufficient(m: _Long) -> str:
    """Too short a record: the slow center that turned back once (geo_centre_max_pass_rev, the Numbers' 'center’s
    largest one-way pass'), else the turning points (geo_n_tp)."""
    row = m.row
    n_tp, passed = row.get('geo_n_tp'), row.get('geo_centre_max_pass_rev')
    if 'centre_turned_insufficient' in set(m.v.rules_fired):
        swings = f' swings ({vt.fmt(n_tp)} turning points), but its slow center' if vt.finite(n_tp) else '’s slow center'
        after = f', after at most {vt.fmt(passed)} of a turn' if vt.finite(passed) and float(passed) < 1 else ''
        return f'{m.noun}{swings} has turned back only once{after} {_panels(L_ANGLE, L_CENTRE)}.'
    if vt.finite(n_tp):
        return f'{m.noun} has only {vt.fmt(n_tp)} turning points in {vt.years(m.span)}, too few to close one cycle {_panels(L_ANGLE, L_CENTRE)}.'
    return f'{m.noun} has too few turning points to close one cycle {_panels(L_ANGLE, L_CENTRE)}.'


def _seen_long_circulation(m: _Long) -> str:
    passed = m.row.get('geo_centre_max_pass_rev')
    runs = f': its center passes {vt.fmt(passed)} turns one way in {vt.years(m.span)}' if vt.finite(passed) and vt.finite(m.span) else ''
    return f'{m.noun} circulates {_panels(L_ANGLE, L_MOD, L_CENTRE)}{runs}.'


SEEN_LONG = {
    'libration': _seen_long_libration,
    'episodic': _seen_long_episodic,
    'circulation': _seen_long_circulation,
    'drift': lambda m: f'{m.noun} oscillates about a center that moves one way {_panels(L_ANGLE, L_CENTRE)}.',
    'near_separatrix': lambda m: f'{m.noun} has no libration: it slips over the top {_panels(L_MOD)}.',
    'insufficient_time': _seen_long_insufficient,
}


def _seen_long(m: _Long) -> str:
    """Sentence 1 of a secular or ZLK row: what the angle does, by the angle word (an unphysical orbit: why)."""
    if m.status == 'invalid' and m.reason == 'unphysical':
        text = vt.unphysical_text(m.a, m.row)
        if '(hyperbolic)' in text:
            return f'The orbit becomes unbound: {text.replace(" (hyperbolic)", ", a hyperbolic orbit")} {_panels(L_E)}.'
        return f'The orbit is unphysical: {text or "see the record"} {_panels(L_E)}.'
    build = SEEN_LONG.get(m.angle)
    return build(m) if build else f'{m.noun} is not defined over the record {_panels(L_ANGLE)}.'


def _long_gap(m: _Long) -> str:
    """'during 825–927 kyr': the longest gap between two librating episodes, where the angle does not librate; ''
    without one."""
    return longest_gap(m.librating)


def _transient_words(m: _Long) -> List[str]:
    """'It is transient, not resonant, because ω does not librate during 825–927 kyr.' for a transient row with a gap
    between its librating episodes (a share of 91% alone reads like a resonant row: 162474)."""
    gap = _long_gap(m) if m.status == 'transient' else ''
    return [f'It is transient, not resonant, because {m.short} does not librate {gap}; a longer record cannot change that.'] if gap else []


def _gate_no_verdict_words(m: _Long) -> str:
    gate, row = m.gate, m.row
    if gate is None:
        return 'the gate did not run'
    if gate.detail == 'coorbital_unfitted_forcing':
        planet = vt._coorbital_planet(row)
        return (
            f'the free-element model does not apply to a co-orbital of {planet}, because the fitted planetary modes do not '
            f'include the forcing of {planet} itself'
        )
    mode = vcc.text(row.get('zlk_gate_node_near'))
    distance, resolution = row.get('zlk_gate_node_near_d_arcsec_yr'), row.get('zlk_gate_node_near_resolution_arcsec_yr')
    if m.outcome == 'planetary_residual' and mode and vt.finite(distance) and vt.finite(resolution):
        return (
            f'the free node rate lies within {vt.fmt(distance)}″/yr of {mode}, less than this record can resolve '
            f'({vt.fmt(resolution)}″/yr), so the free node cannot yet be told from the forced one'
        )
    if m.outcome == 'planetary_residual' and mode:
        return f'the free node cannot yet be told from {mode} in this record'
    return GATE_NO_VERDICT[m.outcome][1] if m.outcome in GATE_NO_VERDICT else f'its outcome is {vt.snake_words(m.outcome) or "missing"}'


FREE_OMEGA_GLOSS = 'ω of the free eccentricity and inclination vectors, the planets’ forced terms removed'
EI_ZLK = 'e and i trade in antiphase at constant √(1−e²) cos i'


def _ei_sentence(m: _Long, first: bool) -> str:
    """The deciding episode's e-i exchange: a ZLK coupling as a positive sentence, the objection of an episode whose
    e and i do not trade so (flag ei_integral_not_held, `verdict_text.episode_ei_parts`: the numbers Checks step 2
    prints); `first`: Lidov–Kozai not yet named in full. '' when not measured or neither."""
    ei = m.episode_ei()
    if ei is None:
        return ''
    name = 'a Lidov–Kozai (ZLK) cycle' if first else 'a ZLK cycle'
    if m.zlk_ei():
        return f'In the deciding episode {EI_ZLK}, as in {name} {_panels(L_E, L_I)}.'
    if vt.has_mark(m.v, 'ei_integral_not_held'):
        parts = vt.episode_ei_parts(ei)
        m.said = True
        return f'But in the deciding episode e and i do not trade as in {name} {_panels(L_E, L_I)}: {parts[0]}, and {parts[2]}.'
    return ''


def _t_long_confirmed(m: _Long) -> List[str]:
    """A confirmed free-ω gate in positive words: the free ω does not wind (or moves at most so many turns per cycle
    inside the episodes), whereas a libration held by the forced eccentricity would leave it winding; 'It is a
    Lidov–Kozai libration' only when the deciding episode's e and i trade as a ZLK cycle (88899: they do not)."""
    row = m.row
    if vcc.text(row.get('zlk_gate_detail')) == 'episodes':
        turns = row.get('zlk_gate_free_turns_episodes_max')
        one = len(m.librating) == 1
        moves = f'moves at most {vt.fmt(turns)} turns per cycle' if vt.finite(turns) else 'does not wind'
        net, extent = row.get('free_omega_revolutions'), row.get('free_omega_range_deg')
        whole = ''
        if vt.finite(net) and vt.finite(extent) and float(extent) >= 360:
            whole = f', although over the whole record the free ω winds ({vt.fmt(abs(float(net)))} net turns)'
        gate = (
            f'Inside {"the episode" if one else "each episode"} the free ω ({FREE_OMEGA_GLOSS}) {moves} {_panels(L_DRIFT)}, '
            f'so {"the libration is" if one else "these librations are"} not imposed by the forced eccentricity, which '
            f'would leave the free ω winding{whole}.'
        )
        return [gate, _ei_sentence(m, first=True)]
    extent, turns = row.get('free_omega_range_deg'), m.free_turns()
    if vt.finite(extent) and float(extent) < 360:  # the Key facts' measure of the free ω, its full range
        stays = f'does not wind; it stays within a {vt.fmt(extent)}° range'
    elif turns:
        stays = f'does not wind: its mean rate |g − s| makes only {turns} turns over the record'
    else:
        stays = 'does not wind'
    lead = 'It is a Lidov–Kozai (ZLK) libration, not one imposed by the planets: ' if m.zlk_ei() else ''
    gate = (
        f'{lead}the free ω ({FREE_OMEGA_GLOSS}) {stays} {_panels(L_DRIFT)}, whereas a libration held by the forced '
        'eccentricity would leave it winding.'
    )
    return [gate, _ei_sentence(m, first=not lead)]


def _t_long_dynamic(m: _Long) -> List[str]:
    if m.zlk and m.outcome == 'confirmed':
        return _transient_words(m) + _t_long_confirmed(m)
    return _t_long_geometric(m)


SECULAR_SHAPE = (
    'For a secular resonance the status rests on the shape of σ alone: there is no force-law test and no search for a '
    'competing resonance, as there is for a mean-motion resonance.'
)


def _secular_phase_words(m: _Long) -> List[str]:
    """The phase-error test that passed (Checks step 2): 'Its largest full swing, 70.3°, is larger than the 0.847° error
    of the g6 phase, so its turning points are σ’s own.'; [] when not compared or not larger."""
    swing, spread = m.row.get('sec_phase_swing_deg'), m.row.get('sec_phase_uncertainty_deg')
    if not (vt.finite(swing) and vt.finite(spread) and float(swing) > float(spread)):
        return []
    modes, _ = vt._phase_modes(m.a)
    names = f' of the {", ".join(modes)} phase{"s" if len(modes) > 1 else ""}' if modes else ' of the planetary phases'
    return [
        f'Its largest full swing, {vt.fmt(swing)}°, is larger than the {vt.fmt(spread)}° error{names}, so its turning points are σ’s own.'
    ]


def _t_long_geometric(m: _Long) -> List[str]:
    if not m.zlk:
        if str(m.v.angle_subtype) == 'phase_unresolved':
            return _phase_words(m)
        return _transient_words(m) + [SECULAR_SHAPE, *_secular_phase_words(m)]
    if m.outcome == 'confirmed':
        return _transient_words(m) + _t_long_confirmed(m)
    out = _transient_words(m)
    objections = objection_checks(m.row)
    measured = [name for name in GATE_OBJECTIONS if name in objections]
    if measured:
        # a measured objection of the gate (`policy._zlk_measured`): the level reads it, the status stays the shape's
        rho = m.row.get('zlk_gate_rho_max')
        below = f' (ρ_max {vt.fmt(rho)})' if pol.OBJECTION_ZLK_FREE_E in measured and vt.finite(rho) else ''
        words = _join([OBJECTION_SEEN[name][0] for name in measured]) + below
        letters = tuple(dict.fromkeys(OBJECTION_SEEN[name][1] for name in measured))
        out.append(f'The free-ω gate measured an objection: {words} {_panels(*letters)}; it holds the level, not the status.')
    else:
        out.append(f'The free-ω gate gives no verdict here: {_gate_no_verdict_words(m)}.')
    m.said = True  # the sentence above names the objection or the missing gate verdict
    if pol.OBJECTION_ZLK_INTEGRAL in objections:
        out.append(_ei_sentence(m, first=True))  # the e-i objection in its own words (it also marks it said)
    if vt.has_mark(m.v, 'kinematic_excluded') and m.gate is not None and vt.finite(m.gate.rho_max):
        ei = f', and in the deciding episode {EI_ZLK}, as in a ZLK cycle {_panels(L_E, L_I)}' if m.zlk_ei() else ''
        out.append(
            f'A forced libration is excluded all the same: the free eccentricity reaches {vt.fmt(m.gate.rho_max)} times '
            f'the forced one, too large for the planets’ forcing to hold the perihelion {_panels(L_VECTOR)}{ei}.'
        )
    out.append('The status therefore follows the shape of ω.')
    return out


def _phase_words(m: _Long) -> List[str]:
    swing, spread = m.row.get('sec_phase_swing_deg'), m.row.get('sec_phase_uncertainty_deg')
    numbers = f' (largest full swing {vt.fmt(swing)}° against {vt.fmt(spread)}°)' if vt.finite(swing) and vt.finite(spread) else ''
    return [
        f'The swing is within how far the linear phases of the planetary modes stray over the record{numbers}: its turning '
        'points may be those of the phases.'
    ]


def _node_words(lock, mode: str) -> str:
    """What the node lock supports about the node and the forcing mode (`zlk.node_lock`, the Evidence row's facts):
    'The node is locked to g5 too' only for a resolved lock, 'The node moves at the rate of g5 as closely as the record
    can resolve' for a rate within 1/T that closes no cycle of Ω − ϖ_j; '' when the lock names no such mode."""
    if lock is None or lock.candidate != mode:
        return ''
    if lock.locked:
        return f'The node is locked to {mode} too'
    if lock.reason == 'unresolved':
        return f'The node moves at the rate of {mode} as closely as the record can resolve'
    return ''


def _attribution_words(m: _Long) -> List[str]:
    """The forced vector behind a kinematic ZLK libration, only from the measured attribution and node lock
    (`zlk.attribution`, `zlk.node_lock`; docs/physics/secular-zlk.md#zlk-free-omega): the forced e larger than the
    free one, the libration period the free–forced beat (the gate sub-check's numbers), the node."""
    zlk = getattr(m.a, 'zlk', None)
    if zlk is None or not zlk.attribution.attributed:
        return []
    attribution = zlk.attribution
    mode = attribution.mode.partition(' ')[0]
    ratio = attribution.free_over_forcing
    if not (vt.finite(ratio) and ratio < 1):
        return []
    beat = ''
    if vt.finite(attribution.beat_period_yr):
        beat = f'; that circling, 2π/|g − {mode}| = {vt.years(attribution.beat_period_yr)}, is the libration period of ω'
    out = [
        f'This is a forced libration, not a Lidov–Kozai one: the forced eccentricity of mode {mode} is larger than the '
        f'free one (free/forced {vt.fmt(ratio)}), so ϖ stays near the forced direction while the free eccentricity '
        f'vector circles it {_panels(L_VECTOR)}{beat}.'
    ]
    node = _node_words(zlk.node_lock, mode)
    if node:
        out.append(f'{node}, so ω = ϖ − Ω stays bounded.')
    return out


def _t_long_kinematic(m: _Long) -> List[str]:
    gate, row = m.gate, m.row
    if m.veto == pol.VETO_ZLK_INTEGRAL or vcc.text(m.v.origin_detail) == 'ei_integral_not_held':
        ei = getattr(gate, 'episode_ei', None) if gate is not None else None
        eta = f' (η_H {vt.fmt(ei.eta_h)})' if ei is not None and vt.finite(ei.eta_h) else ''
        return [
            f'Forced, not Lidov–Kozai: e and i do not trade at constant √(1−e²) cos i in the deciding episode{eta} '
            f'{_panels(L_E, L_I)}, where the forced terms are too small to break that integral.'
        ]
    # one count of the free ω's turns, named by what it is: the net turns of the free elements (free_omega_revolutions,
    # the Key facts' number); the mean-rate count |g − s|·T only when the net one is missing (both: Numbers, FREE-ω GATE)
    turns = m.free_turns()
    winds = f'{turns} turns at its mean rate |g − s| over the record' if turns else ''
    net, extent = row.get('free_omega_revolutions'), row.get('free_omega_range_deg')
    if vt.finite(net) and vt.finite(extent) and float(extent) >= 360:
        winds = f'{vt.fmt(abs(float(net)))} net turns over the record'
    winds = f': {winds}' if winds else ''
    if m.veto == pol.VETO_ZLK_FREE_E:
        return [
            f'Kinematic (forced) libration: bounded by the forced eccentricity, not by Lidov–Kozai dynamics; the free ω '
            f'winds{winds} {_panels(L_DRIFT)}.',
            _free_e_pin_words(row),
        ]
    attribution = _attribution_words(m)
    if attribution:
        return attribution + [f'The free ω, the angle that librates in a true Lidov–Kozai cycle, winds instead{winds} {_panels(L_DRIFT)}.']
    zlk = getattr(m.a, 'zlk', None)
    mode = f' of mode {zlk.attribution.mode.partition(" ")[0]}' if zlk is not None and zlk.attribution.attributed else ''
    return [
        f'Kinematic (forced) libration: bounded by the forced eccentricity{mode}, not by Lidov–Kozai dynamics; the free ω '
        f'winds{winds} {_panels(L_DRIFT)}.'
    ]


def _free_e_pin_words(row: Mapping[str, Any]) -> str:
    """Where the veto zlk_free_e pinned the free e below the forced e (`policy._free_e_pins`): over the whole record
    (ρ_max < 1, zlk_gate_rho_max), else inside every librating episode (ρ_episodes < 1 with the free ω turning at least
    ZLK_EPISODE_FREE_TURNS_PER_CYCLE per cycle; ρ_max ≥ 1 there)."""
    rho = row.get('zlk_gate_rho_max')
    if vt.finite(rho) and float(rho) < 1:
        return (
            f'The free e stays below the forced e over the whole record (at most {vt.fmt(rho)} of it), so ϖ follows the '
            f'forced direction {_panels(L_VECTOR)}.'
        )
    inside, turns = row.get('zlk_gate_rho_episodes'), row.get('zlk_gate_free_turns_episodes')
    below = f' (at most {vt.fmt(inside)} of it)' if vt.finite(inside) else ''
    turning = f' while the free ω turns {vt.fmt(turns)} times per cycle' if vt.finite(turns) else ''
    return (
        f'Inside every librating episode the free e stays below the forced e{below}{turning}, so ϖ follows the forced '
        f'direction there {_panels(L_VECTOR)}.'
    )


def _t_long_circulation(m: _Long) -> List[str]:
    if vcc.text(m.row.get('pol_decided_by')).split(';')[0] == pol.DECIDED_INSUFFICIENT:
        return [_ends_in_circulation(m)]
    if m.zlk:
        return ['There is no libration for the free-ω gate to test, so this is no Lidov–Kozai libration.']
    return ['A circulating σ is not in this resonance; no other test is needed.']


def _ends_in_circulation(m: _Long) -> str:
    """A row the policy decided as too short that ends in circulation (`policy._insufficient`, W36 F5): σ_f's range from
    the last turning point (`policy._end_witness`: of any run or (MMR) candidate span in the window, a trailing stall
    dropped; else the window's start) to the end, `pol_end_travel_rev`."""
    travel, t_last = m.row.get('pol_end_travel_rev'), m.row.get('pol_t_last_tp_yr')
    geometry = m.a.geometry
    start = geometry is not None and vt.finite(t_last) and float(t_last) <= float(geometry.window[0])
    where = 'over the whole window' if start else 'from its last turning point to the end'
    spans = f'{vt.fmt(travel)} turns' if vt.finite(travel) else 'at least one turn'
    return (
        f'But {where} the filtered {m.short} spans {spans}, at least one: the record ends in circulation, so the status is '
        f'decided now {_panels(L_ANGLE)}.'
    )


def _t_long_baseline(m: _Long) -> List[str]:
    if str(m.v.angle_subtype) == 'phase_unresolved':
        return _phase_words(m)
    if 'centre_turned_insufficient' in set(m.v.rules_fired):
        return [
            'That is part of one slow swing of the center: not a drift, since the center turned back, but too short to '
            'tell a slow libration of the center from a slow circulation.'
        ]
    return ['The shape cannot yet tell a libration from a slow circulation.']


def _t_long_unphysical(m: _Long) -> List[str]:
    return [f'The elements of an unbound orbit are not those of an ellipse, so neither {m.short} nor the free elements are read.']


def _t_long_gate_unavailable(m: _Long) -> List[str]:
    m.said = True
    return [f'The free-ω gate gives no verdict: {_gate_no_verdict_words(m)}.']


def _t_long_drift(m: _Long) -> List[str]:
    return ['The center moves one way: there is no fixed center to librate about in this record.']


LONG_REASON_WORDS = {'origin_unknown': 'The cause of the libration is not identified'}  # D9: no 'origin' (secular)


def _t_long_generic(m: _Long) -> List[str]:
    words = LONG_REASON_WORDS.get(m.reason, vt.snake_words(m.reason).capitalize())
    return [f'{words}: the angle is not judged further.']


LONG_TEMPLATES: Dict[str, Callable[[_Long], List[str]]] = {
    Reason.DYNAMIC_LIBRATION: _t_long_dynamic,
    Reason.DYNAMIC_EPISODE: _t_long_dynamic,
    Reason.GEOMETRIC_LIBRATION: _t_long_geometric,
    Reason.GEOMETRIC_EPISODE: _t_long_geometric,
    Reason.KINEMATIC_LIBRATION: _t_long_kinematic,
    Reason.CIRCULATION: _t_long_circulation,
    Reason.BASELINE_NEEDED: _t_long_baseline,
    Reason.UNPHYSICAL: _t_long_unphysical,
    Reason.GATE_UNAVAILABLE: _t_long_gate_unavailable,
    Reason.DRIFT_MECHANISM_UNKNOWN: _t_long_drift,
    Reason.MODULATED_CIRCULATION: _t_long_drift,
    Reason.UNDEFINED_PHASE: _t_long_generic,
    Reason.UNUSABLE_SERIES: _t_long_generic,
    Reason.UNDERSAMPLED: _t_long_generic,
}


def _long_start_gap(m: _Long) -> str:
    """'before 406 kyr': the first librating episode misses the record's start by more than the coverage allows
    (its coverage_start_gap_P, the Numbers' 'gap to the start'), and the record grows forward only."""
    if not m.librating or vcc.text(m.v.extend_direction) not in ('', 'forward'):
        return ''
    first = m.librating[0]
    gap = first.get('coverage_start_gap_P')
    if vt.finite(gap) and float(gap) > m.s.libration_coverage_periods:
        return f'before {vt.years(first.get("t_start_yr"))}'
    return ''


def would_change_long(m: _Long) -> str:
    """The last line (module docstring): what a longer record cannot do (a transient row with a gap between its
    episodes or before the first one), the next check (the extend clause; the free node's separation time), or all of
    it under 'Needed: …'. A resonant row's extension counts more cycles (its N bar), never 'would change'."""
    row, checks = m.row, []
    clause = vcc.text(row.get('pol_extend_clause'))
    to = m.v.extend_to_years if str(m.v.action) == 'extend' else None
    length = f'a record of {vt.years(to)}' if vt.finite(to) else ''
    period = row.get('sec_decisive_period_yr')
    if clause == 'B' and length and vt.finite(period):
        checks.append(f'{length}, since the deciding period ({vt.years(period)}) is too long for this record')
    elif clause in ('A', 'B') and length:
        state, detail, ceiling = m.inputs.get('N', ('', '', ''))
        if m.status == 'resonant' and vt.finite(state) and detail.startswith('<'):
            checks.append(f'{length}, to count more cycles ({vt.fmt(float(state))} now; {vcc._bar(detail, ceiling)})')
        elif m.status == 'transient' and true(row.get('pol_last_reaches_end')):
            checks.append(f'{length}, which shows whether {m.short}, still librating at the end, keeps librating')
        else:
            checks.append(length)
    elif clause == 'C' and length:
        checks.append(f'{length}, long enough for the test that could overrule it')
    separation = row.get('zlk_gate_node_separation_yr')
    mode = vcc.text(row.get('zlk_gate_node_near'))
    if m.zlk and mode and vt.finite(separation):
        checks.append(
            f'a record of at least {vt.years(separation)}, over which the free node separates from {mode}, so the gate can '
            'give its own verdict'
        )
    # a gap between the episodes is said in the Why (`_transient_words`); the start only here
    gap = _long_start_gap(m) if m.status == 'transient' and not _long_gap(m) else ''
    return _last_line(m.status, f'{m.short} does not librate {gap}' if gap else '', (), checks)


def long_why(analysis, settings, row: Mapping[str, Any], episodes: Sequence[Mapping[str, Any]]) -> vc.Why:
    """The Why block of a secular or Lidov-Kozai row: `row` = the summary row with the body's extra columns,
    `episodes` its episodes.csv rows."""
    m = _Long(analysis, settings, row, episodes)
    template = LONG_TEMPLATES.get(m.reason, _t_long_generic)
    meaning = template(m)
    said = m.said or any('objection' in sentence or 'no verdict' in sentence for sentence in meaning)
    sentences = [_seen_long(m), *meaning, limit_sentence({**row, 'reason': m.reason}, vcc.text(m.v.confidence), said)]
    return vc.Why(_fit(sentences), would_change_long(m))
