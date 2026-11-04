"""W29 acceptance of the minimal classifier policy (spec tmp/.../w29/SPEC.md §10): invariants, clear sets, flag rates,
the design-set and held-out checks and the confusion against the author's labels, from finished reclassify runs.

    uv run python benchmark/tools/w29_acceptance.py \
        --minimal v2-w29-minimal --minimal-szlk v2-w29-minimal-secular-zlk \
        --v2 v2-w29-v2 --v2-szlk v2-w29-v2-secular-zlk \
        --baseline v2-w28-head --baseline-szlk v2-w28-head-secular-zlk \
        --answers tmp/classifier-v2-2026-10-15/w29/answers.json --heldout tmp/classifier-v2-2026-10-15/w29/heldout_ids.csv \
        --out tmp/classifier-v2-2026-10-15/w29/acceptance

Writes report.md, invariants.csv, clear.csv, confusion_*.csv, design_set.csv, held_out.csv, flags.csv and
hard_swaps.csv into --out. It reads labels and lists, never edits them. A hard swap's physical reason is read from
--swap-reasons (example_id, reason; written by hand after the run). Exit 0 when the gate holds: every invariant on
every row, every clear family >= 99 % right at >= high with zero opposite verdicts, every hard swap with a reason.

The invariants recompute the policy's level and extend predicate from the `pol_*`, `n_cycles` and
`confidence_basis` columns with the spec's tables written out again here (`level`, `extend_expected`), not imported
from the package: a second reading of §5/§6, so a slip in either shows as a violation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_io as bio  # noqa: E402

FAMILIES = ('coorbital', 'mmr2', 'mmr3', 'secular', 'zlk')
LIBRATING = ('resonant', 'transient', 'near_separatrix')
LEVELS = ('', 'low', 'medium', 'high', 'certain')  # '' = no level (uncertain, invalid)
GEOMETRY_NON_RESONANT = ('circulation', 'modulated_circulation', 'alias', 'no_resonant_potential')
VETOES = ('owner', 'kinematic', 'forced_wave', 'strength', 'zlk_kinematic', 'zlk_free_e', 'zlk_integral')
ZLK_VETOES = ('zlk_kinematic', 'zlk_free_e', 'zlk_integral')
GATE_FLAGS = ('gate_unavailable', 'gate_disabled')
POLICY_FLAGS = (
    'gate_unavailable',
    'gate_disabled',
    'one_coorbital_leg',
    'one_half_libration',
    'rejected_candidate',
    'measured_objection',
    'strength_scale_implausible',
    'veto_thin',
    'veto_needs_length',
    'policy_unknown_leaf',
)
COMPARED = ('status', 'angle_status', 'confidence', 'reason', 'action', 'flags', 'caps')  # invariant 9
NUMBER_COLUMNS = (
    'smp_span_yr',
    'geo_n_tp_cut_half',
    'geo_n_tp_cut_double',
    'jump_times_yr',
    'geo_window_t0_yr',
    'geo_window_t1_yr',
    'ep_n_tp',
)
# §10.7 b: the final runs a pre-label status may come from (plain W runs; never a probe, shadow or partial run)
WHITELIST = re.compile(r'^v2-w\d+[a-z]?(-final)?(-secular-zlk)?$')
CLEAR_MIN_CYCLES = 5.0  # §10.3 (a script constant, not a setting)
CLEAR_MIN_PASS_REV = 3.0
CLEAR_PASS = 0.99
# the d-items of answers.json and their objects (dev/status.md, spec §10.5)
D_ITEMS = {
    'd01': '2016 JA',
    'd02': '33128',
    'd03': '2019 DJ1',
    'd04': '2026 AC4',
    'd05': '2007 VU6',
    'd06': 'the g01-5 leg rows',
    'd07': '2015 WZ12, 2024 AV2, 2024 XP16, 2022 BB2',
    'd08': '2009 BE81 20 kyr',
    'd09': '2023 RS2',
    'd10': '1981 1 Myr',
    'd11': '10563 100 kyr',
    'd12': '5026',
    'd13': 'TK14 1 yr',
    'd14': '17346',
    'd15': '2003 ND 40 kyr',
    'c1': '42946',
}
# the d-items whose chosen option the minimal policy reverses by a principle (spec §10.5; reported, never gating)
REVERSED = {
    'd04': 'P2: a force law that was not measured is a flag, not a status move (transient/medium)',
    'd10': 'PRINCIPLES (1981 Midas resonant <= medium, flag gate_unavailable)',
    'd12': 'PRINCIPLES / P7 (5026 uncertain: swing below the mode-phase uncertainty; phase_unresolved is not non_resonant)',
}


# ------------------------------------------------------------------ small readers


def text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ''
    out = str(value).strip()
    return '' if out == 'nan' else out


def number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def items(value) -> List[str]:
    return [item for item in text(value).split(';') if item]


def family(example_id: str) -> str:
    return text(example_id).split('__', 1)[0]


def three_class(status: str) -> str:
    status = text(status)
    if status in LIBRATING:
        return 'librating'
    if status == 'non_resonant':
        return 'non_resonant'
    return 'uncertain' if status else ''


def opposite(a: str, b: str) -> bool:
    """librating against non_resonant, either way (a hard swap)."""
    return {three_class(a), three_class(b)} == {'librating', 'non_resonant'}


def own_rev(row) -> float:
    """The centre's max pass in the angle's own revolutions: pi for a ZLK omega (stored in 2 pi units, x2)."""
    value = number(row.get('geo_centre_max_pass_rev'))
    return 2 * value if family(row.get('example_id')) == 'zlk' else value


def parse_basis(basis: str) -> Dict[str, str]:
    """`N=<float>(<kind>);M=..;V=..;S=..;veto=<name|->[:thin]` -> {'N', 'N_kind', 'M', 'V', 'S', 'veto', 'veto_thin'}."""
    out: Dict[str, str] = {}
    for part in items(basis):
        key, _, value = part.partition('=')
        if key == 'N':
            match = re.fullmatch(r'([^()]*)(?:\(([^)]*)\))?', value)
            out['N'], out['N_kind'] = (match[1], match[2] or '') if match else (value, '')
        elif key == 'veto':
            name, _, thin = value.partition(':')
            out['veto'], out['veto_thin'] = ('' if name == '-' else name), thin
        else:
            out[key] = value
    return out


# ------------------------------------------------------------------ the spec's tables, written out again

FORCE_CHECKS = ('check_mmr_a_response', 'check_mmr_potential', 'check_mmr_strength')


def level(status: str, basis: Dict[str, str], row, settings) -> str:
    """§6 with the W31 ladder: the level the policy must give `status` with this evidence ('' for uncertain and
    invalid). The decisive episode's own cycles (n_TP - 1) / 2 come from `ep_n_tp` (N when it is not in the row)."""
    high, certain = _cycles(settings)
    medium_n = float(getattr(settings, 'confidence_medium_cycles', 1.5))
    core_n = float(getattr(settings, 'confidence_certain_core_cycles', certain))
    n, v = number(basis.get('N')), basis.get('V', '')
    n = n if math.isfinite(n) else 0.0
    tps = number(row.get('ep_n_tp'))
    core = (tps - 1.0) / 2.0 if math.isfinite(tps) else n
    flags = set(items(row.get('flags')))
    if status in ('resonant', 'transient'):
        if v == 'ambiguous' or n < medium_n:
            return 'low'
        full = basis.get('M') == 'full' and basis.get('S') == 'full' and v == 'clean'
        if n >= certain and core >= core_n and full:
            found = 'certain'
        elif n >= high and v in ('clean', 'partial'):
            found = 'high'
        else:
            found = 'medium'
        # W33: an MMR row with none of L4, R0, F passing is at most CONFIDENCE_GEOMETRY_ONLY_MAX (read from the checks;
        # a row without these columns predates W33)
        recorded = [text(row.get(c)) for c in FORCE_CHECKS if c in row]
        if text(row.get('resonance_type')) in ('', 'mmr') and recorded and 'pass' not in recorded:
            limit = str(getattr(settings, 'confidence_geometry_only_max', 'low'))
            found = min(found, limit, key=LEVELS.index)
        return found
    if status == 'near_separatrix':
        return 'low' if v == 'ambiguous' else 'medium'
    if status == 'non_resonant':
        candidate = bool(items(row.get('candidate_episode_ids')))
        if text(row.get('veto')):
            thin = 'veto_thin' in flags or basis.get('veto_thin') == 'thin'
            # W31 (the author): a measured veto is high; a drawn candidate no longer lowers it
            return 'medium' if thin or 'second_capture_unresolved' in flags else 'high'
        if candidate:
            return 'medium'
        if text(row.get('reason')) != 'modulated_circulation':
            if family(row.get('example_id')) in ('secular', 'zlk'):
                tps = [number(row.get(c)) for c in ('geo_n_tp', 'geo_n_tp_cut_half', 'geo_n_tp_cut_double')]
                if text(row.get('mrg_P')) == 'full' and all(t == 0 for t in tps):
                    return 'certain'
            else:
                # an alias has no C1: C5 (a far from the resonance, the branch unique) measures the circulation (Q18)
                measured = 'mrg_C5' if text(row.get('angle_subtype')) == 'alias' else 'mrg_C1'
                if text(row.get(measured)) == 'full' and text(row.get('mrg_C2')) == 'full':
                    return 'certain'
        return 'high'
    return ''


def extend_expected(row, settings) -> Tuple[bool, str]:
    """§5 recomputed from the `pol_*` columns: (extend needed, the clause that holds)."""
    if text(row.get('status')) == 'invalid' or text(row.get('angle_subtype')) == 'phase_unresolved':
        return False, ''  # §5 action order: secular phase_unresolved -> review, never extend
    if not number(row.get('pol_end_travel_rev')) < 1:  # ends in circulation (or not measured: no extend)
        return False, ''
    high, _ = _cycles(settings)
    clause = text(row.get('pol_extend_clause'))
    if clause == 'A1' and ':end' in text(row.get('geo_leg_read')):
        return True, 'A1'
    if clause == 'A2' and ':end' in text(row.get('geo_half_libration_read')):
        return True, 'A2'
    reaches = text(row.get('pol_last_reaches_end')) in ('True', 'true', '1', '1.0')
    if text(row.get('angle_status')) == 'insufficient_time' and not reaches:
        return True, 'A'
    if reaches and number(row.get('pol_n_last')) < high:
        return True, 'A'
    period, span = number(row.get('pol_period_dec_yr')), span_yr(row)
    if math.isnan(period):
        if number(row.get('geo_n_tp')) == 0:  # an arc or a stall: no turning point, B holds
            return True, 'B'
    elif period > 2 * span / settings.libration_min_turning_points:
        return True, 'B'
    # C: a needed veto not measured for lack of length, the length it needs within the next-check budget (Q5, 2026-11-02)
    need = number(row.get('pol_veto_need_yr'))
    flags = set(items(row.get('flags')))
    if 'veto_needs_length' in flags and math.isfinite(need) and need <= settings.mmr_next_check_span_factor * span:
        return True, 'C'
    return False, ''


def span_yr(row) -> float:
    for column in ('pol_span_yr', 'smp_span_yr'):
        value = number(row.get(column))
        if value > 0:
            return value
    return number(row.get('geo_window_t1_yr')) - number(row.get('geo_window_t0_yr'))


def _cycles(settings) -> Tuple[float, float]:
    return float(getattr(settings, 'confidence_high_cycles', 2)), float(getattr(settings, 'confidence_certain_cycles', 3))


# ------------------------------------------------------------------ invariants (§10.2)


def invariants(frame: pd.DataFrame, episodes: pd.DataFrame, settings) -> pd.DataFrame:
    """One row per violation (invariant, example_id, resonance, detail) of invariants 1-8 on a minimal run's rows."""
    counted = _counted_episodes(episodes)
    high, _ = _cycles(settings)
    members: Dict[str, float] = {}
    for row in frame.to_dict('records'):
        key = text(row.get('example_id'))
        members[key] = max(
            members.get(key, 0.0), number(row.get('n_counted_episodes')) if row.get('reason') != 'subresonance_librates' else 0.0
        )
    out = []
    for row in frame.to_dict('records'):
        status, veto, flags = text(row.get('status')), text(row.get('veto')), set(items(row.get('flags')))
        n_counted = number(row.get('n_counted_episodes'))
        n = number(row.get('n_cycles'))

        def bad(inv: str, detail: str, row=row):
            out.append({'invariant': inv, 'example_id': row.get('example_id'), 'resonance': row.get('resonance'), 'detail': detail})

        if text(row.get('policy')) != 'minimal':
            bad('0', f'policy {text(row.get("policy"))!r}, not minimal')
        if text(row.get('angle_status')) == 'libration' and n >= high and status == 'non_resonant' and not veto:
            bad('1', f'libration with N {n:g} >= {high:g} is non_resonant without a veto')
        member = _member_counts(row, members)
        if own_rev(row) >= 1 and n_counted == 0 and status in ('resonant', 'transient') and not member:
            bad('2', f'centre pass {own_rev(row):.2f} rev, no counted episode, status {status}')
        if status == 'non_resonant' and not veto and text(row.get('reason')) not in GEOMETRY_NON_RESONANT:
            bad('3', f'non_resonant/{text(row.get("reason"))} without a veto')
        if text(row.get('reason')) == 'modulated_circulation' and not own_rev(row) >= 1:
            bad('3', f'modulated_circulation with centre pass {own_rev(row):.2f} rev')
        if veto and veto not in VETOES:
            bad('3', f'unknown veto {veto!r}')
        if veto and not any(item.startswith('policy:') and veto in item for item in items(row.get('decision_path'))):
            bad('3', f'veto {veto} has no policy: ledger entry')
        expected, clause = extend_expected(row, settings)
        if expected != (text(row.get('action')) == 'extend'):
            bad(
                '4',
                f'action {text(row.get("action"))!r}, predicate {expected} ({clause or "-"}, column {text(row.get("pol_extend_clause")) or "-"})',
            )
        basis = parse_basis(text(row.get('confidence_basis')))
        if status in LIBRATING or status == 'non_resonant':
            if not basis:
                bad('5', 'no confidence_basis')
            else:
                want = level(status, basis, row, settings)
                if text(row.get('confidence')) != want:
                    bad('5', f'confidence {text(row.get("confidence"))!r}, recomputed {want!r} from {text(row.get("confidence_basis"))}')
                # the basis prints N to three significant digits
                if not math.isclose(number(basis.get('N')), float(f'{n:.3g}'), rel_tol=1e-6, abs_tol=1e-9) and not (
                    math.isnan(n) and math.isnan(number(basis.get('N')))
                ):
                    bad('5', f'n_cycles {n} differs from the basis N {basis.get("N")}')
        if items(row.get('caps')) or any(text(row.get(c)) for c in row if str(c).startswith('cell_')):
            bad('5', 'caps or cell_* not blank under minimal')
        if status == 'uncertain' and text(row.get('action')) not in ('extend', 'review', 'resample'):
            bad('6', f'uncertain with action {text(row.get("action"))!r}')
        if family(row.get('example_id')) == 'zlk':
            _zlk(row, flags, veto, status, bad)
        if status in ('resonant', 'transient') and not n_counted >= 1 and not member:
            bad('8', f'{status} with {n_counted:g} counted episodes')
        if status == 'non_resonant' and veto and n_counted != 0:
            bad('8', f'vetoed non_resonant with {n_counted:g} counted episodes')
        legs = sum(1 for item in items(row.get('counted_episode_ids')) if item == 'leg')
        csv_counted = counted.get((text(row.get('example_id')), text(row.get('resonance'))), 0)
        if not math.isnan(n_counted) and csv_counted + legs != n_counted:
            bad('8', f'episodes.csv counted {csv_counted} + legs {legs} != n_counted_episodes {n_counted:g}')
    return pd.DataFrame(out, columns=['invariant', 'example_id', 'resonance', 'detail'])


def _member_counts(row, members: Dict[str, float]) -> bool:
    """A rule-1 subresonance parent takes its member's status, level and N (§2.1): its counted episodes are the
    member's (another row of the same example), not its own."""
    return text(row.get('reason')) == 'subresonance_librates' and members.get(text(row.get('example_id')), 0.0) >= 1


def _zlk(row, flags, veto, status, bad) -> None:
    code = text(row.get('status_code'))
    gate_flag = bool(flags & set(GATE_FLAGS))
    # the gate's <= medium is for a bounded angle the gate could not judge; a circulating row does not need the gate;
    # W31 D2: kinematic_excluded (the e-i integral held, rho_max >= ZLK_RHO_KINEMATIC_EXCLUDED) lifts it
    capped = gate_flag and 'kinematic_excluded' not in flags
    if capped and ((status in LIBRATING and text(row.get('confidence')) in ('high', 'certain')) or code == '-2'):
        bad('7', f'gate flag with confidence {text(row.get("confidence"))!r}, code {code}')
    if (veto in ZLK_VETOES) != (code == '-2'):
        bad('7', f'veto {veto!r} against status_code {code}')
    if code == '-9' and status != 'uncertain':
        bad('7', f'code -9 on {status}')
    if status in ('resonant', 'transient'):
        check = text(row.get('check_zlk_free_omega'))
        # §8: a kinematic outcome that pins no episode leaves a transient at low with the measured objection
        objection = check == 'fail' and status == 'transient' and 'measured_objection' in flags
        if (check not in ('pass', 'not_measured') and not objection) or (check == 'not_measured') != gate_flag:
            bad('7', f'{status} with check_zlk_free_omega {check!r}, gate flag {gate_flag}')


def _counted_episodes(episodes: pd.DataFrame) -> Dict[Tuple[str, str], int]:
    if episodes is None or not len(episodes) or 'counted' not in episodes:
        return {}
    flag = episodes['counted'].astype(str).isin(('True', 'true', '1', '1.0'))
    keys = ['example_id', 'resonance'] if 'resonance' in episodes else ['example_id']
    sums = episodes.assign(_c=flag).groupby(keys)['_c'].sum()
    return {(k if isinstance(k, tuple) else (k, '')): int(v) for k, v in sums.items()}


def invariant_9(v2: pd.DataFrame, baseline: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The v2 run against the HEAD baseline: (violations, the S1 rows excused because geo_half_libration_read changed)."""
    key = ['example_id', 'resonance']
    merged = v2.merge(baseline, on=key, how='outer', suffixes=('', '_base'), indicator=True)
    out, excused = [], []
    for row in merged.to_dict('records'):
        if row['_merge'] != 'both':
            out.append(
                {'invariant': '9', 'example_id': row['example_id'], 'resonance': row['resonance'], 'detail': f'only in {row["_merge"]}'}
            )
            continue
        diff = [c for c in COMPARED if c in v2 and text(row.get(c)) != text(row.get(f'{c}_base'))]
        if not diff:
            continue
        detail = '; '.join(f'{c}: {text(row.get(c + "_base"))!r} -> {text(row.get(c))!r}' for c in diff)
        entry = {'invariant': '9', 'example_id': row['example_id'], 'resonance': row['resonance'], 'detail': detail}
        if text(row.get('geo_half_libration_read')) != text(row.get('geo_half_libration_read_base')):
            excused.append(entry)
        else:
            out.append(entry)
    columns = ['invariant', 'example_id', 'resonance', 'detail']
    return pd.DataFrame(out, columns=columns), pd.DataFrame(excused, columns=columns)


# ------------------------------------------------------------------ clear sets (§10.3)


def clear_kind(row) -> str:
    """'librator', 'circulator' or '' by measurements the policy does not read for status or level (§10.3)."""
    fam = family(row.get('example_id'))
    angle, n = text(row.get('angle_status')), number(row.get('n_cycles'))
    if angle == 'libration' and n >= CLEAR_MIN_CYCLES:
        if fam in ('coorbital', 'mmr2', 'mmr3'):
            no_jump = not text(row.get('jump_times_yr')) and not number(row.get('geo_n_episodes_before_jump')) > 0
            physics = text(row.get('mrg_R0')) == 'full' or text(row.get('mrg_L4')) == 'full'
            if no_jump and text(row.get('mrg_L1')) == 'full' and not text(row.get('owner')) and physics:
                return 'librator'
        elif fam == 'zlk' and text(row.get('zlk_gate')) == 'confirmed':
            return 'librator'
        elif fam == 'secular' and (
            text(row.get('check_secular_e_response')) == 'pass' or text(row.get('sec_all_modes_measured')) == 'True'
        ):
            return 'librator'
    tps = [number(row.get(c)) for c in ('geo_n_tp', 'geo_n_tp_cut_half', 'geo_n_tp_cut_double')]
    margin = 'mrg_C1' if fam in ('coorbital', 'mmr2', 'mmr3') else 'mrg_P'
    candidate = (
        items(row.get('candidate_episode_ids'))
        or text(row.get('geo_leg_read'))
        or text(row.get('geo_half_libration_read'))
        or number(row.get('geo_near_episode_n_tp')) > 0
    )
    if own_rev(row) >= CLEAR_MIN_PASS_REV and all(t == 0 for t in tps) and text(row.get(margin)) == 'full' and not candidate:
        return 'circulator'
    return ''


def clear_set(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(the clear rows with their verdicts, per family: n, right at >= high, opposite, share, pass)."""
    rows = []
    for row in frame.to_dict('records'):
        kind = clear_kind(row)
        if not kind:
            continue
        status, confidence = text(row.get('status')), text(row.get('confidence'))
        want = 'librating' if kind == 'librator' else 'non_resonant'
        right = three_class(status) == want and status != 'near_separatrix' and LEVELS.index(confidence or '') >= LEVELS.index('high')
        rows.append(
            {
                'family': family(row['example_id']),
                'example_id': row['example_id'],
                'resonance': row.get('resonance'),
                'kind': kind,
                'status': status,
                'confidence': confidence,
                'right': right,
                'opposite': opposite(status, 'resonant' if kind == 'librator' else 'non_resonant'),
            }
        )
    table = pd.DataFrame(rows, columns=['family', 'example_id', 'resonance', 'kind', 'status', 'confidence', 'right', 'opposite'])
    per = []
    for fam in FAMILIES:
        part = table[table['family'] == fam]
        n = len(part)
        share = float(part['right'].mean()) if n else math.nan
        per.append(
            {
                'family': fam,
                'n': n,
                'right': int(part['right'].sum()),
                'opposite': int(part['opposite'].sum()),
                'share': share,
                'pass': n == 0 or (share >= CLEAR_PASS and int(part['opposite'].sum()) == 0),
            }
        )
    return table, pd.DataFrame(per)


# ------------------------------------------------------------------ flag rates (§10.4)


def flag_rates(minimal: pd.DataFrame, v2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(per family and policy: shares and the SavePlotMode row counts, the top 15 flags per family under minimal)."""
    from resonances.simulation.config import SavePlotMode, status_selected

    rows, top = [], []
    for name, frame in (('minimal', minimal), ('v2', v2)):
        for fam in FAMILIES:
            part = frame[frame['example_id'].map(family) == fam]
            n = len(part)
            if not n:
                continue
            flags = part['flags'].map(items) if 'flags' in part else pd.Series([[]] * n, index=part.index)
            row = {
                'policy': name,
                'family': fam,
                'n': n,
                'uncertain': float((part['status'] == 'uncertain').mean()),
                'action_not_none': float(part['action'].map(text).isin(('', 'none')).eq(False).mean()),
                'review_required': float(part['review_required'].map(text).isin(('True', 'true', '1')).mean()),
                'policy_flag': float(flags.map(lambda f: bool(set(f) & set(POLICY_FLAGS))).mean()),
            }
            for mode in SavePlotMode:
                row[f'mode_{mode.value}'] = int(
                    sum(
                        status_selected(
                            mode,
                            text(r.get('status')) or None,
                            text(r.get('libration_origin')),
                            text(r.get('action')),
                            text(r.get('confidence')),
                        )
                        for r in part.to_dict('records')
                    )
                )
            rows.append(row)
            if name == 'minimal':
                counts = pd.Series([f for fl in flags for f in fl]).value_counts().head(15)
                top += [{'family': fam, 'flag': flag, 'rows': int(k)} for flag, k in counts.items()]
    return pd.DataFrame(rows), pd.DataFrame(top, columns=['family', 'flag', 'rows'])


# ------------------------------------------------------------------ labels (§10.5-10.7)


def verdict_text(row: Optional[dict]) -> str:
    if row is None:
        return 'missing'
    return '/'.join(text(row.get(c)) or '-' for c in ('status', 'angle_status', 'confidence', 'action'))


def design_set(answers: dict, minimal: pd.DataFrame, v2: pd.DataFrame) -> pd.DataFrame:
    """§10.5: every answers.json verdict key with v2 and minimal, the author's verdict and comment verbatim."""
    rows = []
    mins, olds = _catalogue(minimal), _catalogue(v2)
    for key, answer in answers.get('verdicts', {}).items():
        example = key.split('::')[0]
        old, new = olds.get(key, olds.get(example)), mins.get(key, mins.get(example))
        changed = (
            old is not None
            and new is not None
            and (text(old.get('status')), text(old.get('confidence')))
            != (
                text(new.get('status')),
                text(new.get('confidence')),
            )
        )
        rows.append(
            {
                'example_id': key,
                'v2': verdict_text(old),
                'minimal': verdict_text(new),
                'author_verdict': text(answer.get('verdict')),
                'author_comment': text(answer.get('comment')),
                'agree_row_changed': text(answer.get('verdict')) == 'agree' and changed,
            }
        )
    for item, decision in answers.get('decisions', {}).items():
        rows.append(
            {
                'example_id': f'{item}: {D_ITEMS.get(item, "?")}',
                'v2': '',
                'minimal': REVERSED.get(item, ''),
                'author_verdict': f'option {text(decision.get("option"))}',
                'author_comment': text(decision.get('comment')),
                'agree_row_changed': False,
            }
        )
    return pd.DataFrame(rows)


def _catalogue(frame: pd.DataFrame) -> Dict[str, dict]:
    """example_id (and example_id::resonance) -> the row; a catalogue row (no parent) wins its example_id."""
    out: Dict[str, dict] = {}
    for row in frame.to_dict('records'):
        out[f'{row["example_id"]}::{text(row.get("resonance"))}'] = row
        if not text(row.get('parent')) or row['example_id'] not in out:
            out[row['example_id']] = row
    return out


def label_frame(minimal: pd.DataFrame, v2: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """The author's labels next to the minimal and v2 statuses and levels of the same catalogue row."""
    mins, olds = _catalogue(minimal), _catalogue(v2)
    rows = []
    for row in labels.to_dict('records'):
        new, old = mins.get(row['example_id']), olds.get(row['example_id'])
        rows.append(
            {
                **row,
                'minimal_status': text(new.get('status')) if new else '',
                'minimal_confidence': text(new.get('confidence')) if new else '',
                'v2_status': text(old.get('status')) if old else '',
                'v2_confidence': text(old.get('confidence')) if old else '',
            }
        )
    return pd.DataFrame(rows)


def hard_swaps(frame: pd.DataFrame, reasons: Dict[str, str]) -> pd.DataFrame:
    """Author labels at certain/high whose minimal status is the opposite class, with v2's and the recorded reason."""
    strong = frame[frame['author_confidence'].map(text).isin(('certain', 'high'))]
    swapped = strong[[opposite(a, m) for a, m in zip(strong['author_status'], strong['minimal_status'])]].copy()
    swapped['v2_swapped'] = [opposite(a, o) for a, o in zip(swapped['author_status'], swapped['v2_status'])]
    swapped['reason'] = swapped['example_id'].map(lambda e: reasons.get(e, ''))
    columns = [
        'example_id',
        'author_status',
        'author_confidence',
        'minimal_status',
        'minimal_confidence',
        'v2_status',
        'v2_swapped',
        'reason',
    ]
    return swapped[columns].reset_index(drop=True)


def confusion(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return pd.crosstab(frame['author_status'].map(three_class), frame[column].map(three_class).replace('', 'missing'), margins=True)


def held_out(frame: pd.DataFrame) -> Dict[str, float]:
    """§10.6: hard swaps on certain/high author labels and 3-class agreement, minimal and v2."""
    strong = frame[frame['author_confidence'].map(text).isin(('certain', 'high'))]
    out: Dict[str, float] = {'n': len(frame), 'n_certain_high': len(strong)}
    for name in ('minimal', 'v2'):
        out[f'{name}_hard_swaps'] = int(sum(opposite(a, s) for a, s in zip(strong['author_status'], strong[f'{name}_status'])))
        agree = frame['author_status'].map(three_class) == frame[f'{name}_status'].map(three_class)
        out[f'{name}_agreement'] = float(agree.mean()) if len(frame) else math.nan
    out['expectation_holds'] = out['minimal_hard_swaps'] <= out['v2_hard_swaps']
    return out


def label_date(source: str) -> str:
    """The date of an author label: 'author:YYYY-MM-DD' in its source, the export's file date, else the decisions'."""
    found = re.search(r'author:(\d{4}-\d{2}-\d{2})', source) or re.search(r'author_labels_(\d{4}-\d{2}-\d{2})', source)
    if found:
        return found[1]
    return '2026-10-15' if 'AUTHOR-QUESTIONS' in source else ''


def whitelisted_runs(data: Path) -> List[Tuple[str, str, bool]]:
    """(created, name, secular_zlk) of the whitelisted final runs under `data`, oldest first."""
    runs = []
    for folder in sorted(Path(data).iterdir()) if Path(data).is_dir() else []:
        if not WHITELIST.match(folder.name) or not (folder / 'run.json').exists() or not (folder / 'summary.csv').exists():
            continue
        created = text(json.loads((folder / 'run.json').read_text()).get('created'))
        runs.append((created, folder.name, folder.name.endswith('-secular-zlk')))
    return sorted(runs)


def prelabel(labels: pd.DataFrame, data: Path, read=None) -> Tuple[pd.DataFrame, int]:
    """§10.7 b: each label's status from the latest whitelisted run of its family created before the label's date
    (the previous one when the example is missing there). Returns (rows, skipped labels)."""
    read = read or (lambda name: _catalogue(pd.read_csv(Path(data) / name / 'summary.csv', dtype=str, keep_default_na=False)))
    runs, cache, rows, skipped = whitelisted_runs(data), {}, [], 0
    for row in labels.to_dict('records'):
        date, long_period = label_date(text(row.get('label_source'))), family(row['example_id']) in ('secular', 'zlk')
        found = None
        for created, name, szlk in reversed(runs):
            if szlk != long_period or not date or created[:10] >= date:
                continue
            cache.setdefault(name, None)
            if cache[name] is None:
                cache[name] = read(name)
            if row['example_id'] in cache[name]:
                found = (name, text(cache[name][row['example_id']].get('status')))
                break
        if found is None:
            skipped += 1
            continue
        rows.append({**row, 'label_date': date, 'prelabel_run': found[0], 'prelabel_status': found[1]})
    return pd.DataFrame(rows), skipped


# ------------------------------------------------------------------ the run


def read_family_run(mmr: Optional[str], szlk: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(summary, episodes) of an MMR run (with the mmr_numbers columns the checks read) and its secular/ZLK run."""
    frames, episodes = [], []
    for name in (mmr, szlk):
        if not name:
            continue
        folder = Path(name) if Path(name).is_dir() else bio.DATA / name
        summary = pd.read_csv(folder / 'summary.csv', dtype=str, keep_default_na=False)
        numbers = folder / 'mmr_numbers.csv'
        if numbers.exists():
            header = bio.read_header(numbers)
            wanted = ['example_id', 'resonance'] + [c for c in NUMBER_COLUMNS if c in header and c not in summary]
            summary = summary.merge(
                pd.read_csv(numbers, usecols=wanted, dtype=str, keep_default_na=False), on=['example_id', 'resonance'], how='left'
            )
        frames.append(summary)
        if (folder / 'episodes.csv').exists():
            episodes.append(pd.read_csv(folder / 'episodes.csv', dtype=str, keep_default_na=False, low_memory=False))
    summary = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return summary, (pd.concat(episodes, ignore_index=True) if episodes else pd.DataFrame())


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('minimal', 'v2', 'baseline'):
        ap.add_argument(f'--{name}', required=name != 'baseline', help=f'the {name} MMR run (name under benchmark/data or a path)')
        ap.add_argument(f'--{name}-szlk', help=f'the {name} secular/ZLK run')
    ap.add_argument('--answers', type=Path, help='answers.json (the design set, §10.5)')
    ap.add_argument('--heldout', type=Path, help='heldout_ids.csv frozen before policy.py (§10.6)')
    ap.add_argument('--swap-reasons', type=Path, help='CSV example_id,reason: the physical reason of each hard swap')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    return run(args)


def run(args) -> int:
    import reclassify
    from resonances.resonance.analysis.settings import settings_from_config

    settings = settings_from_config()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    minimal, episodes = read_family_run(args.minimal, args.minimal_szlk)
    v2, _ = read_family_run(args.v2, args.v2_szlk)
    found = invariants(minimal, episodes, settings)
    excused = pd.DataFrame()
    if args.baseline:
        base, _ = read_family_run(args.baseline, args.baseline_szlk)
        nine, excused = invariant_9(v2, base)
        found = pd.concat([found, nine], ignore_index=True)
        excused.to_csv(out / 'invariant9_s1_rows.csv', index=False)
    found.to_csv(out / 'invariants.csv', index=False)
    clear_rows, clear_per = clear_set(minimal)
    clear_rows.to_csv(out / 'clear.csv', index=False)
    rates, top = flag_rates(minimal, v2)
    rates.to_csv(out / 'flags.csv', index=False)
    top.to_csv(out / 'flags_top.csv', index=False)
    labels = reclassify.author_labels()
    source = labels['label_source'].map(text)
    claude = ~(source.str.startswith('author_labels_') | source.str.contains(': author:') | source.str.contains(': AUTHOR-QUESTIONS'))
    control = source.str.contains('control', case=False)
    current = label_frame(minimal, v2, labels[~claude & ~control & (labels['author_status'].map(text) != '')])
    for name in ('minimal', 'v2'):
        confusion(current, f'{name}_status').to_csv(out / f'confusion_current_{name}.csv')
    pre, skipped = prelabel(current, bio.DATA)
    if len(pre):
        confusion(pre, 'prelabel_status').to_csv(out / 'confusion_prelabel.csv')
    reasons = {}
    if args.swap_reasons and Path(args.swap_reasons).exists():
        reasons = dict(pd.read_csv(args.swap_reasons, dtype=str, keep_default_na=False)[['example_id', 'reason']].values)
    swaps = hard_swaps(current, reasons)
    swaps.to_csv(out / 'hard_swaps.csv', index=False)
    regressions = current[
        [LEVELS.index(text(m)) < LEVELS.index(text(o)) for m, o in zip(current['minimal_confidence'], current['v2_confidence'])]
    ]
    regressions.to_csv(out / 'confidence_regressions.csv', index=False)
    design = design_set(json.loads(Path(args.answers).read_text()), minimal, v2) if args.answers else pd.DataFrame()
    design.to_csv(out / 'design_set.csv', index=False)
    held = {}
    if args.heldout:
        frozen = pd.read_csv(args.heldout, dtype=str, keep_default_na=False)
        held_rows = label_frame(minimal, v2, frozen)
        held_rows.to_csv(out / 'held_out.csv', index=False)
        held = held_out(held_rows)
    gate = found.empty and bool(clear_per['pass'].all()) and bool((swaps['reason'].map(text) != '').all())
    gate = gate and bool(held.get('expectation_holds', True))  # §10.6: no more held-out hard swaps than v2
    report = [
        '# W29 acceptance',
        '',
        f'runs: minimal {args.minimal} + {args.minimal_szlk}; v2 {args.v2} + {args.v2_szlk}; baseline {args.baseline} + {args.baseline_szlk}',
        f'**gate: {"PASS" if gate else "FAIL"}**',
        '',
        f'## Invariants (10.2): {len(found)} violations on {len(minimal)} rows',
        found.groupby('invariant').size().to_string() if len(found) else 'none',
        f'invariant 9: {len(excused)} rows excused by S1 (invariant9_s1_rows.csv)',
        '',
        '## Clear sets (10.3)',
        clear_per.to_string(index=False),
        '',
        '## Flag rates (10.4)',
        rates.to_string(index=False),
        '',
        '## Held-out (10.6)',
        json.dumps(held, indent=1) if held else 'not given',
        '',
        f'## Labels (10.7): {len(current)} current author labels; Claude-written {int(claude.sum())}, control {int((control & ~claude).sum())} excluded',
        'minimal:',
        confusion(current, 'minimal_status').to_string(),
        'v2:',
        confusion(current, 'v2_status').to_string(),
        f'pre-label: {len(pre)} labels, {skipped} skipped (no whitelisted run before the label)',
        confusion(pre, 'prelabel_status').to_string() if len(pre) else '',
        '',
        f'## Hard swaps on author certain/high: {len(swaps)} ({int((swaps["reason"].map(text) == "").sum())} without a reason)',
        swaps.to_string(index=False) if len(swaps) else 'none',
        '',
        f'## Confidence regressions on author-labelled rows: {len(regressions)} (confidence_regressions.csv)',
        '',
        f'## Design set (10.5): {int(design["agree_row_changed"].sum()) if len(design) else 0} "agree" rows changed (design_set.csv)',
    ]
    (out / 'report.md').write_text('\n'.join(report) + '\n')
    print('\n'.join(report))
    return 0 if gate else 1


if __name__ == '__main__':
    sys.exit(main())
