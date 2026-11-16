"""The 'all' form of the right-hand column of the one-page combined figures (V1, plot option `combined.numbers='all'`,
`draw_text`): every number of the decision, shared by the MMR figure (`mmr_verdict_panel`, `mmr_combined`) and the
secular/Lidov-Kozai one (`secular_plots`); also the cell formatting and the column words (`format_cell`,
`COLUMN_LABELS`, `RANGE_LABELS`) the explained column (`verdict_column_*`) prints with.

The text is built from CSV rows only (summary.csv and the type's numbers) and the decision ledger, so the figure and
the CSV always print the same number: each `NumberLine` keeps the column its value came from. A block (`BlockSpec`)
lists its known columns in a fixed order and then every other column of its prefixes, grouped under bold headers.
Numbers are written in CSV units; the unit is read from the column suffix (`_deg`, `_au_yr`, ...) or from a unit
token before a qualifier (`_deg_min`, `_rev_raw`).

- a pure duplicate (the same quantity under another prefix with the same printed value) is printed once, at its
  first place; the kept line lists the other columns in `NumberLine.also`;
- a long list cell is printed as its count, first and last value;
- floats: 4 significant digits in fixed notation from 1e-3 to 1e4, whole numbers above, `1.234e-05` below;
  semi-major axes in au with 7 digits.

Placement: one stream in block order flowing through the text columns of the layout (a block cut by a column end
continues under '<TITLE> (cont.)'); only when every column is full is the font reduced (down to
`VerdictFigureLayout.text_min_font_pt`). Nothing is cut.
"""

from __future__ import annotations

import fnmatch
import math
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from resonances.resonance.analysis.ledger import LedgerEntry

from .style import (
    MARK_COLOURS,
    MONO_CHAR_WIDTH,
    MONO_FONT,
    TEXT_GROUP,
    TEXT_HEADER_FILL,
    TEXT_KEY,
    TEXT_MISSING,
    VerdictFigureLayout,
)

MISSING = '—'
NOT_COMPUTED = 'not computed'
MARK_PASS, MARK_THIN, MARK_FAIL, MARK_NONE = '✓', '~', '✗', '–'

KEY_WIDTH = 15  # characters reserved for a label before its value (a longer label pushes its value right)
PAIR_KEY_WIDTH = 15  # the same in a cell of a two-per-row line: values of both kinds of rows start in one column
BLOCK_GAP = 0.4  # empty space after a block, in lines
FONT_STEP_PT = 0.25  # the text font is reduced in these steps until the columns hold everything
LIST_MAX_ITEMS = 6  # a longer ';' list prints as 'n values: first … last'
CONTINUED = ' (cont.)'

# The verdict block of every resonance type (Verdict.to_flat_dict)
VERDICT_COLUMNS = (
    'status',
    'angle_status',
    'angle_subtype',
    'final_regime',
    'final_regime_start_yr',
    'confidence',
    'confidence_raw',
    'reason',
    'flags',
    'action',
    'extend_to_years',
    'resample_step_yr',
    'libration_origin',
    'origin_detail',
    'owner',
    'neighbours',
    'resonant_argument',
    'cause_role',
    'cause_term',
    'commensurability',
    'multiplet_status',
    'status_code',
)
# Columns printed elsewhere (path, margins) or not at all (rules_fired: the review folder's column, `analysis.rules`;
# review_required/review_reasons: the review queue, which repeats the confidence, caps and failed support shown)
SPECIAL_COLUMNS = ('decision_path', 'thin_margins', 'failed_support', 'caps', 'rules_fired', 'review_required', 'review_reasons')
# Blocks whose lines are never dropped as duplicates (they are the verdict and its comparisons)
NO_DEDUPE = ('verdict', 'path', 'checks', 'margins')


@dataclass(frozen=True)
class NumberLine:
    """One printed line of the text columns: label, value (formatted), unit, its CSV column and a mark.

    A line with an empty label is free text (a ledger entry, a table row, a reason). `group` is the
    bold group header the line is printed under; `also` lists columns of the same value not printed again."""

    block: str
    label: str
    value: str
    unit: str = ''
    column: str = ''
    mark: str = ''
    group: str = ''
    also: Tuple[str, ...] = ()
    style: str = ''  # 'bold' (the decision compared it) | 'grey' (its test was skipped)


@dataclass
class TextBlock:
    name: str
    title: str
    lines: List[NumberLine] = field(default_factory=list)


@dataclass(frozen=True)
class BlockSpec:
    title: str
    columns: Tuple[str, ...] = ()  # known columns, in print order (printed when present in the rows)
    prefixes: Tuple[str, ...] = ()  # every other column with these prefixes follows
    always: Tuple[str, ...] = ()  # printed even when absent from the rows (as '—')
    groups: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()  # (header, fnmatch patterns of columns), first match wins


# Unit of a column from its suffix, longest suffix first
_UNITS = (
    ('_rad_yr_au', 'rad/yr/au'),
    ('_rad_kyr2', 'rad/kyr²'),
    ('_arcsec_yr', '″/yr'),
    ('_deg_yr', '°/yr'),
    ('_rad_kyr', 'rad/kyr'),
    ('_rad_yr', 'rad/yr'),
    ('_au_yr', 'au/yr'),
    ('_per_kyr', '/kyr'),
    ('_deg', '°'),
    ('_rad', 'rad'),
    ('_rev', 'rev'),
    ('_au', 'au'),
    ('_kyr', 'kyr'),
    ('_Myr', 'Myr'),
    ('_yr', 'yr'),
    ('_P', 'P'),
)
# A unit token before a trailing qualifier (`smp_step_rotation_deg_min`, `geo_net_rev_raw`)
_QUALIFIERS = ('min', 'max', 'lo', 'hi', 'med', 'raw', 'filtered', 'predictive', 'numpy', 'pred', 'meas')
_INNER_UNITS = (('deg', '°'), ('rev', 'rev'), ('revs', 'rev'), ('yr', 'yr'), ('kyr', 'kyr'), ('au', 'au'))
_PRECISE_SUFFIX = '_au'  # semi-major axes need 7 digits: a* and a_res differ in the 5th
_PRECISE_DIGITS = 7
_DIGITS = 4
_SMALL = 1e-3  # below: 1.234e-05
_LARGE = 1e4  # at and above: a whole number


# --------------------------------------------------------------------------------------------- cells


def missing(value: Any) -> bool:
    """None, NaN and empty strings are 'not measured' (an empty CSV cell)."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def _inner_unit(column: str) -> Tuple[str, str]:
    """(unit, token) of a unit written before a trailing qualifier, else ('', '')."""
    parts = column.split('_')
    if len(parts) >= 3 and parts[-1] in _QUALIFIERS:
        for token, unit in _INNER_UNITS:
            if parts[-2] == token:
                return unit, token
    return '', ''


def unit_of(column: str) -> str:
    for suffix, unit in _UNITS:
        if column.endswith(suffix):
            return unit
    return _inner_unit(column)[0]


# W28 E11 (card g10-7, author C): the pendulum numbers in words, with the source of the number in the label (observed:
# measured on the angle; 'from observed cycles' / 'from T0': inferred from the observed cycles through the cosine
# pendulum, T0 = P/F(A); 'theory' / 'strength': predicted from the planet's mass by `strength`). T0 is never 'theory'.
# The unit is still read from the column suffix and printed after the value.
COLUMN_LABELS = {
    'ep_period_med_yr': 'observed cycle period P',
    'ep_T0_min_kyr': 'small-swing period T0 min (from observed cycles)',
    'ep_T0_max_kyr': 'small-swing period T0 max (from observed cycles)',
    'ep_omega0_rad_yr': 'small-swing frequency ω0 (from observed cycles)',
    'ep_half_width_au': 'well half-width in a (from T0)',
    'ep_mu_eff': 'strength needed μ_eff (effective, not a mass)',
    'str_mu_eff_req': 'strength needed μ_eff (effective, not a mass)',
    'str_P_lib_theory_kyr': 'theory period (strength)',
    'str_P_lib_theory_min_kyr': 'theory period min (strength)',
    'str_P_lib_theory_max_kyr': 'theory period max (strength)',
    'str_omega0_theory_rad_yr': 'theory frequency ω0 (strength)',
    'str_mu_eff_theory_min': 'theory strength μ_th min',
    'str_mu_eff_theory_max': 'theory strength μ_th max',
    'str_ratio': 'needed / theory strength',
    # h = (E − V_min)/(V_max − V_min) of the fitted V_E (`analysis.pendulum`): 1 is its top, the cosine separatrix only
    # when V_E has one harmonic
    'sep_h_med': 'cycle energy h (1 = top of the fitted V)',
    'sep_zone_h_lo': 'separatrix layer in h, lo',
    'sep_zone_h_hi': 'separatrix layer in h, hi',
    'sep_VE_R2': 'energy fit R²',
    'rsp_coorb_K_over_K_theory': 'force scale / planet’s own mass K/K_th',
}
# Two columns of one quantity printed as one line 'lo–hi unit' in the explained column (`verdict_column_mmr`)
RANGE_LABELS = {
    ('ep_T0_min_kyr', 'ep_T0_max_kyr'): 'small-swing period T0 (from observed cycles)',
    ('str_P_lib_theory_min_kyr', 'str_P_lib_theory_max_kyr'): 'theory period (strength)',
    ('str_mu_eff_theory_min', 'str_mu_eff_theory_max'): 'theory strength μ_th',
    ('sep_zone_h_lo', 'sep_zone_h_hi'): 'separatrix layer in h',
}
# The labels of the redesigned right column (`verdict_column`, 2026-11-04: human words, no column stems), for the
# columns COLUMN_LABELS does not name. Kept apart from COLUMN_LABELS so that the 'all' form keeps its labels (its
# tests read the stems); `column_words` reads COLUMN_LABELS first, then this table.
COLUMN_WORDS = {
    # the deciding episode and the whole record's shape
    'ep_centre_mod_2pi_deg': 'center',
    'ep_n_tp': 'turning points',
    'ep_half_amp_med_deg': 'half-amplitude, median',
    'ep_half_amp_max_deg': 'half-amplitude, max',
    'ep_centre_shift_over_amp': 'center shift / half-amplitude',
    'ep_coverage_start_gap_P': 'gap to the record’s start',
    'ep_coverage_end_gap_P': 'gap to the record’s end',
    'ep_onset_ratio': 'a-wave before / inside the episode',
    'geo_n_tp': 'turning points, whole record',
    'geo_period_cycle_yr': 'cycle period, whole record',
    'geo_centre_max_pass_rev': 'largest pass of the center',
    'geo_drift_fraction': 'drifting share of the record',
    'geo_n_revolutions': 'revolutions of σ',
    'geo_n_slips': 'slips over the top',
    'geo_centre_range_rad': 'range of the slow center',
    'geo_slow_swing_identity_slope_ratio': 'slow swing: |slope/c − 1|',
    'geo_slow_swing_identity_corr': 'slow swing: correlation of σ̇ and a',
    'geo_slow_swing_a_ratio': 'slow swing: a swing / the pendulum’s',
    'geo_slow_swing_period_ratio': 'slow swing: period / T0_th F(A)',
    'geo_leg_read': 'the leg’s discrepancy',
    # the response of a
    'rsp_identity_slope_ratio': 'σ̇ against a: slope / c',
    'rsp_identity_corr': 'σ̇ against a: correlation',
    'rsp_b1_ratio': 'a swing / the swing σ needs',
    'rsp_b1_meas_au': 'a swing',
    'rsp_force_window_yr': 'force window',
    'rsp_force_n_windows': 'force windows',
    'rsp_force_R2_h1': 'force law R², first half',
    'rsp_force_R2_h2': 'force law R², second half',
    'rsp_force_phi0_deg': 'force phase φ0',
    'rsp_phi0_max_dev_deg': 'spread of φ0',
    'rsp_phi0_tol_deg': 'allowed spread of φ0',
    'rsp_coorb_K_over_K_theory_h1': 'force scale K/K_th, first half',
    'rsp_coorb_K_over_K_theory_h2': 'force scale K/K_th, second half',
    'rsp_coorb_energy_R2': '1:1 energy integral R²',
    'rsp_omega0_ratio': 'small-swing frequency / the cycle’s',
    'rsp_b5_ratio': 'slower libration test ratio',
    'rsp_slow_adot_R2': 'slow ȧ on sin σ_slow R²',
    'rsp_response_cutoff_yr': 'response cutoff',
    # pendulum and energy
    'sep_period_law_med': 'period law, median',
    'sep_VE_VF_agree_fraction': 'windows with V_E ≈ V_F',
    'sep_zone_method': 'zone measured by',
    'sep_mechanism': 'what switches the regimes',
    # the resonance
    'ctx_order': 'order',
    'ctx_a_res_nominal_au': 'a_res (catalog)',
    'ctx_a_star_au': 'a* (from the rates)',
    'ctx_a_mean_au': 'mean a',
    'ctx_delta_a_au': 'mean a − a*',
    'ctx_a_from_mean_longitude_rate_au': 'a from the mean-longitude rate',
    'ctx_c_rad_yr_au': 'c = dσ̇/da',
    'ctx_rate_expected_deg_yr': 'σ̇ expected at mean a',
    'ctx_rate_measured_deg_yr': 'σ̇ measured',
    'ctx_P_orb_yr': 'orbital period',
    # kinematics and the slow scale
    'kin_corr_rate': 'σ̇ against p ϖ̇: correlation',
    'kin_e_free': 'free e',
    'kin_e_forced': 'forced e',
    'kin_modulation_period_kyr': 'e-cycle 2π/|g − g5|',
    'kin_two_omega_period_yr': '2ω cycle π/|ω̇|',
    'kin_pred_swing_deg': 'p ϖ half-swing, predicted',
    'kin_meas_swing_deg': 'p ϖ half-swing, measured',
    'slow_period_yr': 'slow period',
    'slow_halfamp_deg': 'slow half-amplitude',
    'slow_cutoff_yr': 'slow-scale cutoff',
    'slow_secular_period_yr': 'nearest secular period',
    'slow_cause_level': 'slow cause',
    # step, alias, filter
    'smp_span_yr': 'record length',
    'smp_step_yr': 'output step',
    'smp_step_rotation_deg_max': 'σ turn per step, max',
    'smp_far_ratio': 'distance of mean a from a*, in a-scales (half its swing)',
    'smp_da_sp_over_a_excursion': 'short-period scatter of a / a swing',
    'flt_cutoff_yr': 'filter cutoff',
    'flt_gain_at_pcyc': 'filter gain at the cycle period',
    # members and the other resonance
    'sub_n_scanned': 'members scanned',
    'sub_best': 'closest member',
    'sub_accepted': 'member that holds the center',
    'own_name': 'driver',
    'own_n_candidates': 'candidate drivers',
    'own_psi_turns_per_cycle': 'turns of Ψ per cycle',
    'own_R2': 'driver’s force law R²',
    'own_cover': 'share of σ’s episode the driver covers',
    'own_lever_gain': 'driver’s phase gain over σ’s',
    'str_ratio': 'needed / theory strength',
    'str_divisor_residual': 'needed / scale × B² (small divisors)',
    'bw_outcome': 'counted episodes the background carries',
}


def column_words(column: str) -> str:
    """The human label of a column for the redesigned column: COLUMN_LABELS, then COLUMN_WORDS; '' when neither names it
    (the caller then leaves the number out of a group, never prints the stem)."""
    return COLUMN_LABELS.get(column) or COLUMN_WORDS.get(column, '')


def label_of(column: str, prefixes: Sequence[str] = ()) -> str:
    """The label of a column: its words from COLUMN_LABELS when it has them, else the column without its block prefix
    (one of `prefixes`) and unit (`smp_unwrap_resid_p99_deg` -> `unwrap_resid_p99`, `smp_step_rotation_deg_min` ->
    `step_rotation_min`)."""
    if column in COLUMN_LABELS:
        return COLUMN_LABELS[column]
    label = column
    for prefix in prefixes:
        if label.startswith(prefix) and len(label) > len(prefix):
            label = label[len(prefix) :]
            break
    for suffix, _ in _UNITS:
        if label.endswith(suffix) and len(label) > len(suffix):
            return label[: -len(suffix)]
    unit, token = _inner_unit(column)
    if unit == '°':  # 'deg' is the unit only; 'rev' stays a word of the label (net_rev_raw)
        parts = label.split('_')
        if len(parts) >= 3 and parts[-2] == token:
            return '_'.join(parts[:-2] + parts[-1:])
    return label


def _list_items(text: str) -> List[str]:
    return [item.strip() for item in text.split(';') if item.strip()]


def format_cell(column: str, value: Any) -> str:
    """A CSV cell as printed: '—' when empty, integers as integers, floats with 4 significant digits
    (7 for semi-major axes, whole numbers from 1e4, `1.234e-05` below 1e-3), lists with '; ' and a
    list longer than LIST_MAX_ITEMS as 'n values: first … last'."""
    if missing(value):
        return MISSING
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if hasattr(value, 'item') and not isinstance(value, str):
        value = value.item()
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _format_float(column, value)
    text = str(value).strip()
    if text in ('True', 'False'):
        return 'yes' if text == 'True' else 'no'
    items = _list_items(text)
    if len(items) > LIST_MAX_ITEMS:
        return f'{len(items)} values: {items[0]} … {items[-1]}'
    return '; '.join(items) if ';' in text else text


def _format_float(column: str, value: float) -> str:
    if math.isinf(value):
        return '∞' if value > 0 else '−∞'
    if value == 0:
        return '0'
    if value.is_integer() and abs(value) < 1e7:
        return str(int(value))
    size = abs(value)
    if column.endswith(_PRECISE_SUFFIX) and size >= 0.1:
        return f'{value:.{_PRECISE_DIGITS}g}'
    if size >= _LARGE:
        return f'{value:.0f}'
    if size < _SMALL:
        return f'{value:.{_DIGITS - 1}e}'
    return f'{value:.{_DIGITS}g}'


def unit_space(column: str) -> str:
    unit = unit_of(column)
    return f' {unit}' if unit else ''


def cell_with_unit(column: str, value: Any) -> str:
    """A cell and its unit for the right column's Numbers: a time in yr of 10⁴ yr and more in kyr (Myr from 10⁷ yr),
    so that neighboring rows read in one unit ('64.41 kyr' beside 'e-cycle 28.47 kyr', not '64409 yr'); the value
    is the cell's, only rescaled."""
    if unit_of(column) == 'yr' and not missing(value) and not isinstance(value, (bool, str)):
        size = abs(float(value))
        if math.isfinite(size) and size >= _LARGE:
            scale, unit = (1e6, 'Myr') if size >= 1e7 else (1e3, 'kyr')
            return f'{float(value) / scale:.{_DIGITS}g} {unit}'
    return format_cell(column, value) + unit_space(column)


# --------------------------------------------------------------------------------------------- blocks


def ledger_marks(decision_path: Sequence[LedgerEntry]) -> Dict[str, str]:
    """Column -> mark of the last ledger entry that compared it."""
    return {entry.column: entry.mark for entry in decision_path if entry.column}


def block_prefixes(specs: Mapping[str, BlockSpec], extra: Sequence[str] = ()) -> Tuple[str, ...]:
    """The prefixes `label_of` strips: those of the blocks, then `extra`."""
    return tuple(prefix for spec in specs.values() for prefix in spec.prefixes) + tuple(extra)


def claimed_columns(specs: Mapping[str, BlockSpec]) -> set:
    return {column for spec in specs.values() for column in spec.columns}


def owner_block(column: str, specs: Mapping[str, BlockSpec], claimed: set) -> Optional[str]:
    """The block a column belongs to: by explicit listing first, then by prefix."""
    for name, spec in specs.items():
        if column in spec.columns:
            return name
    if column in claimed:
        return None
    for name, spec in specs.items():
        if spec.prefixes and column.startswith(spec.prefixes):
            return name
    return None


def group_of(spec: BlockSpec, column: str) -> str:
    for title, patterns in spec.groups:
        if any(fnmatch.fnmatchcase(column, pattern) for pattern in patterns):
            return title
    return 'other' if spec.groups else ''


def number_line(
    block: str, column: str, row: Mapping[str, Any], marks: Mapping[str, str], group: str = '', prefixes: Sequence[str] = ()
) -> NumberLine:
    value = row.get(column)
    return NumberLine(
        block,
        label_of(column, prefixes),
        format_cell(column, value),
        '' if missing(value) else unit_of(column),
        column,
        marks.get(column, ''),
        group,
    )


def column_lines(
    name: str,
    spec: BlockSpec,
    row: Mapping[str, Any],
    marks,
    claimed,
    specs: Mapping[str, BlockSpec],
    prefixes: Sequence[str] = (),
    block_check: Optional[Mapping[str, str]] = None,
) -> List[NumberLine]:
    """The value lines of one block: its known columns, then every other column of its prefixes, grouped. A block
    named in `block_check` (block -> check column) prints nothing when all its columns are empty."""
    if name in ('checks', 'margins', 'path'):
        return []
    columns = [column for column in spec.columns if column in row or column in spec.always]
    columns += [column for column in row if column not in spec.columns and spec.prefixes and owner_block(column, specs, claimed) == name]
    if name in (block_check or {}) and all(missing(row.get(column)) for column in columns):
        return []
    lines = [number_line(name, column, row, marks, group_of(spec, column), prefixes) for column in columns]
    if not spec.groups:
        return lines
    order = [title for title, _ in spec.groups] + ['other']  # lines grouped, the column order kept inside a group
    return sorted(lines, key=lambda line: order.index(line.group))


def other_lines(
    row: Mapping[str, Any], marks, claimed, specs: Mapping[str, BlockSpec], prefixes: Sequence[str] = (), block: str = 'record'
) -> List[NumberLine]:
    """Columns of no block (identity extras, comments): printed in `block`, never dropped."""
    lines = []
    for column in row:
        if column in SPECIAL_COLUMNS or column.startswith(('mrg_', 'check_')) or owner_block(column, specs, claimed):
            continue
        lines.append(number_line(block, column, row, marks, 'other', prefixes))
    return lines


def _stem(column: str, alias_prefixes: Sequence[str]) -> str:
    for prefix in alias_prefixes:
        if column.startswith(prefix):
            return column[len(prefix) :]
    return column


def drop_duplicates(
    blocks: Dict[str, TextBlock], order: Sequence[str], alias_prefixes: Sequence[str], no_dedupe: Sequence[str] = NO_DEDUPE
) -> None:
    """Print a quantity once: lines with the same stem (column without its namespace in `alias_prefixes`), printed
    value and unit are one quantity. The record-level line is kept (an `ep_*` copy of the decisive episode only when
    nothing else prints it), else the first in print order; the kept line records the others in `also`. Empty values
    and the blocks of `no_dedupe` are never merged."""
    seen: Dict[Tuple[str, str, str], List[Tuple[str, int]]] = {}
    for name in [name for name in order if name in blocks]:
        for index, line in enumerate(blocks[name].lines):
            if line.column and line.label and line.value != MISSING:
                seen.setdefault((_stem(line.column, alias_prefixes), line.value, line.unit), []).append((name, index))
    dropped: Dict[str, set] = {name: set() for name in blocks}
    for places in seen.values():
        if len(places) < 2:
            continue
        protected = [place for place in places if place[0] in no_dedupe]
        record = [place for place in places if not blocks[place[0]].lines[place[1]].column.startswith('ep_')]
        keep = (protected or record or places)[0]
        others = [place for place in places if place != keep and place[0] not in no_dedupe]
        if not others:
            continue
        name, index = keep
        kept = blocks[name].lines[index]
        also = tuple(blocks[other[0]].lines[other[1]].column for other in others)
        blocks[name].lines[index] = NumberLine(**{**kept.__dict__, 'also': kept.also + also})
        for other in others:
            dropped[other[0]].add(other[1])
    for name, indices in dropped.items():
        if indices:
            blocks[name].lines = [line for i, line in enumerate(blocks[name].lines) if i not in indices]


def path_lines(decision_path: Sequence[LedgerEntry]) -> List[NumberLine]:
    return [NumberLine('path', '', entry.describe(), column=entry.column, mark=entry.mark) for entry in decision_path]


CHECK_MARKS = {'pass': MARK_PASS, 'fail': MARK_FAIL}
MARGIN_MARKS = {'full': MARK_PASS, 'thin': MARK_THIN, 'fail': MARK_FAIL}


def check_lines(row: Mapping[str, Any], checks: Sequence[str], prefix: str) -> List[NumberLine]:
    """One line per check column of the type (`checks`), labeled without `prefix`."""
    lines = []
    for column in checks:
        value = row.get(column)
        text = MISSING if missing(value) else str(value)
        label = column[len(prefix) :] if column.startswith(prefix) else column
        lines.append(NumberLine('checks', label, text, column=column, mark=CHECK_MARKS.get(text, MARK_NONE)))
    return lines


def margin_ids(row: Mapping[str, Any], order: Sequence[str] = ()) -> List[str]:
    ids = [key[len('mrg_') :] for key in row if key.startswith('mrg_') and not key.endswith('_value')]
    known = [margin for margin in order if margin in ids]
    return known + sorted(margin for margin in ids if margin not in order)


def margin_lines(row: Mapping[str, Any], order: Sequence[str] = ()) -> List[NumberLine]:
    lines = []
    for margin in margin_ids(row, order):
        state = row.get(f'mrg_{margin}')
        state = '' if missing(state) else str(state)
        column = f'mrg_{margin}_value'
        label = f'{margin} {state}' if state else margin
        lines.append(
            NumberLine('margins', label, format_cell(column, row.get(column)), column=column, mark=MARGIN_MARKS.get(state, MARK_NONE))
        )
    for column in ('thin_margins', 'failed_support', 'caps'):
        lines.append(NumberLine('margins', column, format_cell(column, row.get(column)), column=column))
    return lines


def build_blocks(
    row: Mapping[str, Any],
    specs: Mapping[str, BlockSpec],
    decision_path: Sequence[LedgerEntry],
    checks: Sequence[str],
    check_prefix: str,
    alias_prefixes: Sequence[str],
    margin_order: Sequence[str] = (),
    label_prefixes: Sequence[str] = (),
    empty_reason: Optional[Callable[[str, Mapping[str, Any]], str]] = None,
    extra: Optional[Mapping[str, List[NumberLine]]] = None,
) -> Dict[str, TextBlock]:
    """Every block of `specs` (in their order) from one CSV row and the ledger: the verdict, the decision path, the
    checks, the margins, then the value blocks; `extra` lines are appended to their block (tables). Columns of no
    block go to 'record' when there is one. Pure duplicates are printed once; an empty block says why."""
    prefixes = tuple(label_prefixes) or block_prefixes(specs)
    marks = ledger_marks(decision_path)
    claimed = claimed_columns(specs)
    blocks = {name: TextBlock(name, spec.title) for name, spec in specs.items()}
    for name, spec in specs.items():
        blocks[name].lines.extend(column_lines(name, spec, row, marks, claimed, specs, prefixes))
    if 'path' in blocks:
        blocks['path'].lines = path_lines(decision_path)
    if 'checks' in blocks:
        blocks['checks'].lines = check_lines(row, checks, check_prefix)
    if 'margins' in blocks:
        blocks['margins'].lines = margin_lines(row, margin_order)
    for name, lines in (extra or {}).items():
        blocks[name].lines += list(lines)
    if 'record' in blocks:
        blocks['record'].lines += other_lines(row, marks, claimed, specs, prefixes)
    drop_duplicates(blocks, list(specs), alias_prefixes)
    for name, block in blocks.items():
        if not block.lines:
            block.lines.append(NumberLine(name, '', empty_reason(name, row) if empty_reason else NOT_COMPUTED, mark=MARK_NONE))
    return blocks


# --------------------------------------------------------------------------------------------- layout


@dataclass(frozen=True)
class Cell:
    mark: str
    label: str
    value: str  # value and unit
    missing: bool


@dataclass(frozen=True)
class Row:
    """A printed row: 'title', 'group' (a bold group header), 'kv' (one cell), 'pair' (two cells),
    'text' (free text at `indent`) or 'gap'."""

    kind: str
    cells: Tuple[Cell, ...] = ()
    text: str = ''
    indent: int = 0
    mark: str = ''
    missing: bool = False
    height: float = 1.0  # in lines
    block: str = ''  # title of the block the row belongs to (for '(cont.)' headers)
    group: str = ''  # the group header the row is printed under (repeated at the top of a new column)


@dataclass
class Placement:
    font_pt: float
    chars: int  # characters per column line
    capacity: float  # lines per column
    columns: List[List[Row]]
    overflow: float  # lines that did not fit (only at the smallest font)


def _wrap(text: str, width: int) -> List[str]:
    return textwrap.wrap(text, width=max(width, 8), break_long_words=True, break_on_hyphens=False) or ['']


def _cell(line: NumberLine) -> Cell:
    value = f'{line.value} {line.unit}' if line.unit else line.value
    return Cell(line.mark, line.label, value, line.value == MISSING)


def _pair_value_offset(cell: Cell) -> int:
    return 2 + max(len(cell.label) + 1, PAIR_KEY_WIDTH)


def _short(cell: Cell, half: int) -> bool:
    """A cell goes two per row when its label fits the key width and its value the half line."""
    return len(cell.label) <= PAIR_KEY_WIDTH - 1 and _pair_value_offset(cell) + len(cell.value) <= half - 1


def _kv_rows(cell: Cell, chars: int) -> List[Row]:
    if len(cell.label) <= KEY_WIDTH - 1:
        start = 2 + KEY_WIDTH
    else:
        start = 2 + len(cell.label) + 1
    if chars - start < 10:
        rows = [Row('kv', (Cell(cell.mark, cell.label, '', cell.missing),))]
        start = 2 + KEY_WIDTH
        return rows + [Row('text', text=piece, indent=start, missing=cell.missing) for piece in _wrap(cell.value, chars - start)]
    pieces = _wrap(cell.value, chars - start)
    rows = [Row('kv', (Cell(cell.mark, cell.label, pieces[0], cell.missing),))]
    return rows + [Row('text', text=piece, indent=start, missing=cell.missing) for piece in pieces[1:]]


def _text_rows(line: NumberLine, chars: int) -> List[Row]:
    if len(line.value) <= chars - 2:  # a table row keeps its spaces
        return [Row('text', text=line.value, indent=2, mark=line.mark, missing=line.mark == MARK_NONE and not line.column)]
    pieces = _wrap(line.value, chars - 2)
    rows = [Row('text', text=pieces[0], indent=2, mark=line.mark, missing=line.mark == MARK_NONE and not line.column)]
    rest = ' '.join(pieces[1:])
    return rows + [Row('text', text=piece, indent=4) for piece in (_wrap(rest, chars - 4) if rest else [])]


def block_rows(block: TextBlock, chars: int) -> List[Row]:
    """The rows of one block at `chars` characters per line: title, group headers, values (short ones two
    per row), gap."""
    rows = [Row('title', text=block.title)]
    half = chars // 2
    pending: Optional[Cell] = None
    group = ''

    def flush():
        nonlocal pending
        if pending:
            rows.extend(_kv_rows(pending, chars))
            pending = None

    for line in block.lines:
        if line.group and line.group != group:
            flush()
            group = line.group
            rows.append(Row('group', text=group))
        if not line.label:
            flush()
            rows.extend(_text_rows(line, chars))
            continue
        cell = _cell(line)
        if not _short(cell, half):
            flush()
            rows.extend(_kv_rows(cell, chars))
        elif pending:
            rows.append(Row('pair', (pending, cell)))
            pending = None
        else:
            pending = cell
    flush()
    rows.append(Row('gap', height=BLOCK_GAP))
    tagged, group = [], ''
    for row in rows:
        group = row.text if row.kind == 'group' else ('' if row.kind == 'title' else group)
        tagged.append(Row(**{**row.__dict__, 'block': block.title, 'group': group}))
    return tagged


def _geometry(layout: VerdictFigureLayout, font_pt: float) -> Tuple[int, float]:
    width = layout.text_columns[0][2]
    height = layout.text_columns[0][3]
    chars = int(width * 72 / (MONO_CHAR_WIDTH * font_pt))
    capacity = height * 72 / (font_pt * layout.text_line_height)
    return chars, capacity


def _fits_header(queue: List[Row], used: float, capacity: float) -> bool:
    """A title or a group header never ends a column: it moves on with its first row."""
    if queue[0].kind not in ('title', 'group') or len(queue) < 2:
        return True
    need = queue[0].height + queue[1].height
    if queue[1].kind == 'group' and len(queue) > 2:
        need += queue[2].height
    return used + need <= capacity + 1e-9


def _fill(rows: List[Row], n_columns: int, capacity: float) -> Tuple[List[List[Row]], List[Row]]:
    """Fill the columns from one stream; a column that starts inside a block repeats its title + ' (cont.)'."""
    queue = list(rows)
    columns = []
    for _ in range(n_columns):
        while queue and queue[0].kind == 'gap':
            queue.pop(0)
        column, used = [], 0.0
        if queue and queue[0].kind != 'title':
            first = queue[0]
            heads = [Row('title', text=first.block + CONTINUED, block=first.block)]
            if first.group and first.kind != 'group':
                heads.append(Row('group', text=first.group, block=first.block, group=first.group))
            column += heads
            used += sum(head.height for head in heads)
        while queue and used + queue[0].height <= capacity + 1e-9 and _fits_header(queue, used, capacity):
            row = queue.pop(0)
            column.append(row)
            used += row.height
        columns.append(column)
    return columns, queue


def plan_columns(blocks: Dict[str, TextBlock], layout: VerdictFigureLayout, order: Optional[Sequence[str]] = None) -> Placement:
    """The rows of every column at the largest font (from `text_font_pt` down) that holds everything; blocks in
    `order` (default: the order of `blocks`)."""
    order = list(blocks) if order is None else order
    font = layout.text_font_pt
    n_columns = len(layout.text_columns)
    while True:
        chars, capacity = _geometry(layout, font)
        stream = [row for name in order if name in blocks for row in block_rows(blocks[name], chars)]
        columns, rest = _fill(stream, n_columns, capacity)
        overflow = sum(row.height for row in rest if row.kind != 'gap')
        if not overflow or font - FONT_STEP_PT < layout.text_min_font_pt - 1e-9:
            if overflow:  # nothing is cut: the remainder goes below the last column
                columns[-1] += rest
            return Placement(font, chars, capacity, columns, overflow)
        font -= FONT_STEP_PT


def _stats(placement: Placement) -> Dict[str, int]:
    stats = {f'column_{i + 1}': int(sum(row.height for row in column) + 1e-9) for i, column in enumerate(placement.columns)}
    stats['capacity'] = int(placement.capacity)
    stats['overflow'] = int(math.ceil(placement.overflow))
    stats['font_pt'] = placement.font_pt
    return stats


# --------------------------------------------------------------------------------------------- drawing


class _Writer:
    """Writes monospaced rows into one column rectangle of the figure (inches -> figure fraction)."""

    def __init__(self, fig, rect, placement: Placement, layout: VerdictFigureLayout):
        self.fig = fig
        self.x, y, self.width, height = rect
        self.top = y + height
        self.font = placement.font_pt
        self.char = MONO_CHAR_WIDTH * self.font / 72
        self.line = self.font * layout.text_line_height / 72
        self.size = fig.get_size_inches()

    def text(self, col: float, y: float, text: str, **kwargs):
        kwargs.setdefault('color', '#111111')
        self.fig.text(
            (self.x + col * self.char) / self.size[0],
            y / self.size[1],
            text,
            family=MONO_FONT,
            fontsize=kwargs.pop('fontsize', self.font),
            va='top',
            ha='left',
            **kwargs,
        )

    def mark(self, col: float, y: float, mark: str):
        if mark:
            self.text(col, y, mark, color=MARK_COLOURS.get(mark, TEXT_MISSING), weight='bold')

    def title(self, y: float, text: str):
        from matplotlib.patches import Rectangle

        box = Rectangle(
            (self.x / self.size[0], (y - self.line * 1.1) / self.size[1]),
            self.width / self.size[0],
            self.line * 1.15 / self.size[1],
            transform=self.fig.transFigure,
            color=TEXT_HEADER_FILL,
            lw=0,
            zorder=0,
        )
        self.fig.add_artist(box)
        self.text(0.3, y, text, weight='bold', color='#1a1a1a')

    def cell(self, col: float, y: float, cell: Cell, value_col: Optional[float] = None):
        self.mark(col, y, cell.mark)
        self.text(col + 2, y, cell.label, color=TEXT_KEY)
        if cell.value:
            start = value_col if value_col is not None else col + _pair_value_offset(cell)
            self.text(start, y, cell.value, color=TEXT_MISSING if cell.missing else '#111111')

    def row(self, y: float, row: Row, chars: int):
        if row.kind == 'title':
            self.title(y, row.text)
        elif row.kind == 'group':
            self.text(1, y, row.text, weight='bold', color=TEXT_GROUP)
        elif row.kind == 'kv':
            cell = row.cells[0]
            value_col = 2 + KEY_WIDTH if len(cell.label) <= KEY_WIDTH - 1 else None
            self.cell(0, y, cell, value_col)
        elif row.kind == 'pair':
            self.cell(0, y, row.cells[0])
            self.cell(chars // 2, y, row.cells[1])
        elif row.kind == 'text':
            self.mark(0, y, row.mark)
            self.text(row.indent, y, row.text, color=TEXT_MISSING if row.missing else '#111111')


def draw_text(fig, blocks: Dict[str, TextBlock], layout: VerdictFigureLayout, order: Optional[Sequence[str]] = None) -> Dict[str, int]:
    """The text columns; returns `place_blocks` statistics."""
    placement = plan_columns(blocks, layout, order)
    for rect, rows in zip(layout.text_columns, placement.columns):
        writer = _Writer(fig, rect, placement, layout)
        y = writer.top
        for row in rows:
            writer.row(y, row, placement.chars)
            y -= row.height * writer.line
    return _stats(placement)
