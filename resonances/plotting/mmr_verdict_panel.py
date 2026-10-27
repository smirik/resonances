"""The right-hand side of the combined v2 MMR figure, in two forms (`combined_numbers`, plot option
`combined.numbers`).

Explained (default, layout A′): `draw_explained` = the explanation of `verdict_text.explain_mmr` and the numbers of
`explained_lines` (episodes, `EXPLAINED_GROUPS` most important first, the characteristic periods of `period_lines`).

All: every number of the decision as text, below. The engine (cells, units, block assembly, column placement,
drawing) is shared with the secular/ZLK figure in `combined_numbers`; this module holds the MMR blocks and tables.
The text is built from the CSV rows only (the summary.csv row, the mmr_numbers.csv row, episodes.csv and
constants.csv rows) and the decision ledger, so the figure and the CSV always print the same
number: each `NumberLine` keeps the column its value came from. A block lists its known columns in
a fixed order and then every other column of its prefixes (`BLOCKS`), so a column added to a
`to_columns()` later appears without touching this file. Inside a block, columns are grouped under
bold group headers (`BlockSpec.groups`). Numbers are written in CSV units; the unit is read from the
column suffix (`_deg`, `_au_yr`, ...) or from a unit token before a qualifier (`_deg_min`, `_rev_raw`).

What is printed and how:
- block order `TEXT_BLOCKS`: verdict, decision path, checks, margins, subresonances and owner,
  record / step / filter, constants, resonance, geometry, episodes, response, energy / zone,
  kinematics, strength, orbit;
- a pure duplicate (the same quantity under another prefix, e.g. `ep_force_R2_h1` = `rsp_force_R2_h1`,
  `flt_window_t0_yr` = `geo_window_t0_yr`, with the same printed value) is printed once, at its
  first place; the kept line lists the other columns in `NumberLine.also`;
- a long list cell (`geo_centre_tp_times_yr`) is printed as its count, first and last value;
- floats: 4 significant digits in fixed notation from 1e-3 to 1e4, whole numbers above, `1.234e-05`
  below; semi-major axes in au with 7 digits.

Placement: one stream in `TEXT_BLOCKS` order flowing through the three monospaced columns (a block
cut by a column end continues under '<TITLE> (cont.)'); only when the three columns are full is the
font reduced (down to `VerdictFigureLayout.text_min_font_pt`). Nothing is cut.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from resonances.resonance.analysis.ledger import LedgerEntry
from resonances.resonance.analysis.verdict import MMR_CHECKS

from . import combined_numbers as numbers
from .combined_numbers import (  # noqa: F401  (re-exported: the API of this module)
    BLOCK_GAP,
    CONTINUED,
    FONT_STEP_PT,
    KEY_WIDTH,
    LIST_MAX_ITEMS,
    MARK_FAIL,
    MARK_NONE,
    MARK_PASS,
    MARK_THIN,
    MISSING,
    NOT_COMPUTED,
    PAIR_KEY_WIDTH,
    VERDICT_COLUMNS,
    BlockSpec,
    Cell,
    NumberLine,
    Placement,
    Row,
    TextBlock,
    block_rows,
    format_cell,
    missing,
    unit_of,
)
from .style import VerdictFigureLayout
from .verdict_text import episode_accepted, explain_mmr, fmt, physics_columns, years

NOT_RUN_NO_TRIGGER = 'not run: no trigger'
NOT_RUN_NO_DATA = 'not run: data missing'
NOT_RUN_DERIVED = 'not run: derived row (depth 1)'

# Right-hand text: one stream through the three columns, in this order
TEXT_BLOCKS = (
    'verdict',
    'path',
    'checks',
    'margins',
    'subresonances',
    'owner',
    'record',
    'sampling',
    'constants',
    'resonance',
    'geometry',
    'episodes',
    'response',
    'energy',
    'kinematics',
    'strength',
    'orbit',
)

EPISODE_ROWS_MAX = 8  # runs listed one by one; the rest of the rejected ones are counted


MARGIN_ORDER = (
    'L1',
    'L2',
    'L3',
    'L4',
    'L5',
    'L6',
    'L7',
    'L8',
    'O1',
    'Z',
    'S',
    'W',
    'C1',
    'C2',
    'C3',
    'C4',
    'C5',
    'K1',
    'B5',
    'R0',
    'F',
    'U',
)
ORBIT_COLUMNS = (
    'ctx_e_min',
    'ctx_e_med',
    'ctx_e_max',
    'ctx_inc_min_deg',
    'ctx_inc_med_deg',
    'ctx_inc_max_deg',
    'ctx_q_min_au',
    'ctx_Q_max_au',
    'ctx_P_orb_yr',
    'ctx_crosses',
    'ctx_fair_step_fraction',
)

BLOCKS: Dict[str, BlockSpec] = {
    'verdict': BlockSpec('VERDICT', VERDICT_COLUMNS, always=VERDICT_COLUMNS),
    'path': BlockSpec('DECISION PATH (ledger)'),
    'checks': BlockSpec('CHECKS check_mmr_*', MMR_CHECKS, always=MMR_CHECKS),
    'margins': BlockSpec('MARGINS'),
    'subresonances': BlockSpec(
        'SUBRESONANCES',
        ('sub_triggers', 'sub_support', 'sub_n_scanned', 'sub_n_kept', 'sub_accepted', 'sub_accepted_j', 'sub_best'),
        ('sub_',),
        groups=(
            ('scan', ('sub_triggers', 'sub_support', 'sub_n_*', 'sub_window*', 'sub_flags')),
            ('accepted', ('sub_accepted*',)),
            ('best candidate', ('sub_best*',)),
            ('centre of sigma0', ('sub_centre0*', 'sub_sigma0*')),
            ('multiplet theory', ('sub_*',)),
        ),
    ),
    'owner': BlockSpec(
        'OWNER',
        ('own_name', 'own_best', 'own_n_candidates', 'own_R2'),
        ('own_',),
        groups=(
            ('search', ('own_name', 'own_best', 'own_n_candidates', 'own_detail', 'own_flags')),
            ('candidate shown', ('own_R2*', 'own_phi0*', 'own_K*', 'own_strength*', 'own_rate*')),
            ('remainder psi and frequencies', ('own_psi*', 'own_sigma*', 'own_freq*')),
        ),
    ),
    'record': BlockSpec(
        'RECORD',
        (
            'name',
            'resonance',
            'resonance_type',
            'added_by',
            'parent',
            'smp_span_yr',
            'smp_step_yr',
            'smp_n_rows',
            'base_required_kyr',
            'settings_hash',
            'confidence_cells_sha',
            'package_commit',
        ),
        ('base_', 'rec_'),
        always=('name', 'resonance', 'smp_span_yr', 'smp_step_yr', 'smp_n_rows', 'settings_hash', 'confidence_cells_sha'),
        groups=(
            ('identity', ('name', 'resonance', 'resonance_type', 'added_by', 'parent', 'smp_span_yr', 'smp_step_yr', 'smp_n_rows')),
            ('baseline', ('base_*',)),
            ('reproducibility', ('settings_hash', 'confidence_cells_sha', 'package_commit', 'a0', 'e0', 'inc0', 'Omega0', 'omega0', 'M0')),
        ),
    ),
    'sampling': BlockSpec(
        'STEP, ALIAS, UNWRAP, FILTER',
        (
            'smp_verdict',
            'smp_nyquist_deg_yr',
            'smp_step_rotation_deg_min',
            'smp_step_rotation_deg_max',
            'smp_branches',
            'smp_branch_margin_turns',
            'flt_cutoff_yr',
            'flt_cutoff_binding',
            'flt_cutoff_synodic_bound_yr',
            'flt_cutoff_floor_bound_yr',
            'flt_samples_per_cutoff',
        ),
        ('smp_', 'flt_'),
        always=('flt_cutoff_yr',),
        groups=(
            (
                'unwrap',
                ('smp_unwrap*', 'smp_revs_*', 'smp_*suspect*', 'smp_jitter*', 'smp_alias_np*', 'smp_angle_recon*', 'smp_undefined*'),
            ),
            ('filter', ('flt_*',)),
            ('step and alias', ('smp_*',)),
        ),
    ),
    'constants': BlockSpec('CONSTANTS', ('ctx_n_deg_yr', 'ctx_P_syn_yr'), ('gi_',), always=('ctx_P_syn_yr',)),
    'resonance': BlockSpec(
        'RESONANCE',
        (
            'ctx_k',
            'ctx_p',
            'ctx_order',
            'ctx_a_res_nominal_au',
            'ctx_a_star_au',
            'ctx_a_mean_au',
            'ctx_a_p01_au',
            'ctx_a_p99_au',
            'ctx_delta_a_au',
            'ctx_a_from_mean_longitude_rate_au',
            'ctx_a_from_lamdot_au',
            'ctx_c_rad_yr_au',
            'ctx_rate_expected_deg_yr',
            'ctx_rate_measured_deg_yr',
        ),
        ('ctx_',),
        groups=(
            ('coefficients', ('ctx_k', 'ctx_p', 'ctx_order')),
            ('semi-major axis', ('ctx_a_*', 'ctx_delta_a*', 'ctx_c_*')),
            ('rates', ('ctx_rate*', 'ctx_*varpi*', 'ctx_slow*')),
        ),
    ),
    'geometry': BlockSpec(
        'GEOMETRY',
        (
            'geo_n_tp',
            'geo_n_tp_cut_half',
            'geo_n_tp_cut_double',
            'geo_tp_prominence_deg',
            'geo_period_cycle_yr',
            'geo_period_min_yr',
            'geo_period_max_yr',
            'geo_centre_window_yr',
            'geo_centre_max_pass_rev',
            'geo_centre_net_rev',
            'geo_centre_range_rad',
            'geo_centre_n_tp',
            'geo_net_rev_raw',
            'geo_net_rev_filtered',
            'geo_bands_overlap',
        ),
        ('geo_', 'stall_', 'exit_', 'entry_', 'jump_', 'kick_', 'regime_'),
        groups=(
            ('turning points', ('geo_n_tp*', 'geo_tp_*', 'geo_period_*', 'geo_episode_period*', 'geo_window_*')),
            ('runs', ('geo_n_runs', 'geo_n_episodes', 'geo_n_librating*', 'geo_drift*')),
            ('centre', ('geo_centre_*', 'geo_net_rev*', 'geo_band*', 'geo_wiggle*')),
            ('rate', ('geo_sigmadot*', 'geo_rate_modulation', 'geo_rate_a_corr')),
            ('slips and revolutions', ('geo_*slip*', 'geo_n_revolutions', 'geo_h_rev*')),
            ('stall, arc, pattern B', ('stall_*', 'geo_stall*', 'geo_arc*', 'geo_pattern_b*', 'geo_rate_min*')),
            ('exit and entry', ('exit_*', 'entry_*')),
            ('jumps of a', ('jump_*', 'kick_*', 'regime_*')),
        ),
    ),
    'episodes': BlockSpec(
        'EPISODES',
        (),
        ('ep_',),
        groups=(
            ('decisive episode', ('ep_episode_id', 'ep_accepted', 'ep_subtype', 'ep_reject_reason', 'ep_t_*', 'ep_n_tp*', 'ep_period*')),
            ('amplitude and centre', ('ep_half_amp*', 'ep_centre*', 'ep_bands*')),
            ('coverage and window', ('ep_coverage*', 'ep_before*', 'ep_cutoff*', 'ep_window*')),
            ('pendulum', ('ep_T0*', 'ep_omega0*', 'ep_half_width*', 'ep_a_*', 'ep_mu_eff', 'ep_h_*', 'ep_J_*', 'ep_zone*')),
            ('force law and identity', ('ep_force*', 'ep_phi0*', 'ep_identity*', 'ep_b1*')),
            ('origin', ('ep_libration_origin', 'ep_owner')),
        ),
    ),
    'response': BlockSpec(
        'RESPONSE OF a, FORCE LAW',
        (
            'rsp_identity_slope_ratio',
            'rsp_identity_corr',
            'rsp_identity_wave_corr',
            'rsp_identity_amp_ratio',
            'rsp_force_window_yr',
            'rsp_force_R2_h1',
            'rsp_force_R2_h2',
            'rsp_force_law',
            'rsp_force_K_au_yr',
            'rsp_force_phi0_deg',
            'rsp_phi0_coherence',
            'rsp_omega0_K_rad_yr',
            'rsp_omega0_ratio',
            'rsp_b1_req_au',
            'rsp_b1_meas_au',
            'rsp_b1_ratio',
        ),
        ('rsp_',),
        groups=(
            ('identity', ('rsp_identity*',)),
            ('force law', ('rsp_force*', 'rsp_phi0*', 'rsp_omega0*', 'rsp_response_cutoff*')),
            ('B1 excursion of a', ('rsp_b1*',)),
            ('B5 and slow a', ('rsp_b5*', 'rsp_centre_rate*', 'rsp_slow*', 'rsp_forced_wave*', 'rsp_circ*', 'rsp_exact*')),
        ),
    ),
    'energy': BlockSpec(
        'ENERGY / SEPARATRIX ZONE',
        ('sep_N', 'sep_h_med', 'sep_h_p10', 'sep_h_p90', 'sep_sigma_h', 'sep_zone_method', 'sep_zone_h_lo', 'sep_zone_h_hi'),
        ('sep_',),
        groups=(
            ('energy per cycle', ('sep_N', 'sep_h_*', 'sep_sigma_h', 'sep_n_cycles', 'sep_period_law*')),
            ('potential', ('sep_V*', 'sep_Omega*', 'sep_well*', 'sep_top*', 'sep_eps', 'sep_lambda', 'sep_overlap*', 'sep_eq_*')),
            ('zone', ('sep_zone*', 'sep_n_*', 'sep_crossing*', 'sep_e_trend*')),
            ('R0 and mechanism', ('sep_R0*', 'sep_permutation*', 'sep_mechanism', 'sep_separatrix*')),
            ('modulation', ('sep_modulation*', 'sep_cycle*')),
        ),
    ),
    'kinematics': BlockSpec(
        'KINEMATICS',
        (
            'kin_corr_rate',
            'kin_corr_detrended',
            'kin_cancel_ratio',
            'kin_lambda_part_rate_rad_kyr',
            'kin_lambda_part_resid_deg',
            'kin_p_varpi_rate_rad_kyr',
            'kin_e_free',
            'kin_e_forced',
            'kin_modulation_period_kyr',
            'kin_two_omega_period_yr',
            'kin_e_cos2omega_corr',
            'kin_detail',
        ),
        ('kin_',),
    ),
    'strength': BlockSpec(
        'STRENGTH',
        (
            'str_mu_scale',
            'str_mu_scale_no_eq',
            'str_eq_legal',
            'str_omega0_theory_rad_yr',
            'str_P_lib_theory_kyr',
            'str_mu_eff_req',
            'str_ratio',
        ),
        ('str_',),
    ),
    'orbit': BlockSpec('ORBIT: e, i, q, Q', ORBIT_COLUMNS),
}

# The check whose value says why a block is empty
_BLOCK_CHECK = {'subresonances': 'check_mmr_subresonances', 'owner': 'check_mmr_owner'}
# Namespaces under which one quantity is written twice (the decisive episode, the window, ...)
_ALIAS_PREFIXES = ('ep_', 'rsp_', 'sep_', 'geo_', 'flt_', 'ctx_', 'str_', 'smp_', 'kin_', 'base_')

_PREFIXES = numbers.block_prefixes(BLOCKS, ('ctx_',))


# --------------------------------------------------------------------------------------------- cells


def label_of(column: str) -> str:
    """The column without its MMR block prefix and unit (`smp_unwrap_resid_p99_deg` -> `unwrap_resid_p99`)."""
    return numbers.label_of(column, _PREFIXES)


# --------------------------------------------------------------------------------------------- blocks


def text_blocks(
    summary: Mapping[str, Any],
    numbers_row: Mapping[str, Any],
    decision_path: Sequence[LedgerEntry],
    episodes: Sequence[Mapping[str, Any]] = (),
    constants: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, TextBlock]:
    """Every block of the right-hand side from the summary.csv row, the mmr_numbers.csv row and the ledger
    (`analysis.summary_row()`, `analysis.numbers_row()`, `analysis.verdict.decision_path`): values by column name,
    empty cells printed as '—', every ledger entry printed with `LedgerEntry.describe()`.

    `episodes` (episodes.csv rows) and `constants` (constants.csv rows) add the episode and planet tables.
    A number that a ledger entry compared carries that entry's mark. Pure duplicates are printed once.
    """
    row = {**numbers_row, **summary}
    marks = numbers.ledger_marks(decision_path)
    claimed = numbers.claimed_columns(BLOCKS)
    blocks = {name: TextBlock(name, spec.title) for name, spec in BLOCKS.items()}
    for name, spec in BLOCKS.items():
        blocks[name].lines.extend(numbers.column_lines(name, spec, row, marks, claimed, BLOCKS, _PREFIXES, _BLOCK_CHECK))
    blocks['path'].lines = numbers.path_lines(decision_path)
    blocks['checks'].lines = numbers.check_lines(row, MMR_CHECKS, 'check_mmr_')
    blocks['margins'].lines = numbers.margin_lines(row, MARGIN_ORDER)
    table, shown = _episode_lines(episodes, row.get('ep_episode_id'))
    blocks['episodes'].lines = table + [line for line in blocks['episodes'].lines if line.column not in shown]
    blocks['constants'].lines += _constant_lines(constants)
    blocks['record'].lines += numbers.other_lines(row, marks, claimed, BLOCKS, _PREFIXES)
    numbers.drop_duplicates(blocks, TEXT_BLOCKS, _ALIAS_PREFIXES)
    for name, block in blocks.items():
        if not block.lines:
            block.lines.append(NumberLine(name, '', _empty_reason(name, row), mark=MARK_NONE))
    return blocks


def _empty_reason(name: str, row: Mapping[str, Any]) -> str:
    if name in ('subresonances', 'owner') and not missing(row.get('parent')) and str(row.get('parent')):
        return NOT_RUN_DERIVED
    check = _BLOCK_CHECK.get(name)
    value = '' if check is None or missing(row.get(check)) else str(row.get(check))
    if value == 'not_applicable':
        return NOT_RUN_NO_TRIGGER
    if value == 'not_measured':
        return NOT_RUN_NO_DATA
    return NOT_COMPUTED


# Episode table: (episodes.csv column, the ep_* column of the decisive episode, template)
_EPISODE_FIELDS = (
    ('n_tp', 'ep_n_tp', '{} TP'),
    ('period_med_yr', 'ep_period_med_yr', 'P {} yr'),
    ('half_amp_min_deg', 'ep_half_amp_min_deg', 'A_min {}°'),
    ('half_amp_max_deg', 'ep_half_amp_max_deg', 'A_max {}°'),
    ('centre_shift_over_amp', 'ep_centre_shift_over_amp', 'shift/A {}'),
)
# ep_* columns the table line of the decisive episode prints (when the values agree)
_EPISODE_IDENTITY = (
    ('episode_id', 'ep_episode_id'),
    ('accepted', 'ep_accepted'),
    ('subtype', 'ep_subtype'),
    ('reject_reason', 'ep_reject_reason'),
    ('t_start_yr', 'ep_t_start_yr'),
    ('t_end_yr', 'ep_t_end_yr'),
)


def _forced_wave(episode: Mapping[str, Any]) -> bool:
    """An accepted run judged a forced wave of another argument (F6, episodes.csv `forced_wave`: its route, '' for any
    other run): no episode of the angle."""
    value = episode.get('forced_wave')
    return not missing(value) and str(value).strip().lower() not in ('', 'false', '0', 'none')


def _episode_text(episode: Mapping[str, Any]) -> str:
    span = f"{format_cell('t_start_yr', episode.get('t_start_yr'))}–{format_cell('t_end_yr', episode.get('t_end_yr'))} yr"
    parts = [
        template.format(format_cell(column, episode[column])) for column, _, template in _EPISODE_FIELDS if not missing(episode.get(column))
    ]
    text = f"#{format_cell('episode_id', episode.get('episode_id'))} {span} " + ' '.join(parts)
    if episode_accepted(episode):
        return text + (' forced wave' if _forced_wave(episode) else f" {format_cell('subtype', episode.get('subtype'))}")
    return text + f" rejected: {format_cell('reject_reason', episode.get('reject_reason'))}"


def _episode_lines(episodes: Sequence[Mapping[str, Any]], decisive_id: Any = None) -> Tuple[List[NumberLine], set]:
    """One line per run of turning points (episodes.csv): accepted ✓, rejected – with the reason; past
    EPISODE_ROWS_MAX runs the accepted ones, the first and the last are listed and the rest counted.
    Returns the lines and the ep_* columns the decisive episode's line already prints."""
    runs = list(episodes)
    if len(runs) > EPISODE_ROWS_MAX:
        keep = {0, len(runs) - 1} | {i for i, episode in enumerate(runs) if episode_accepted(episode)}
        keep |= {i for i, episode in enumerate(runs) if _same(episode.get('episode_id'), decisive_id)}
        listed = sorted(keep)[:EPISODE_ROWS_MAX] if len(keep) > EPISODE_ROWS_MAX else sorted(keep)
    else:
        listed = list(range(len(runs)))
    lines, shown = [], set()
    for i in listed:
        episode = runs[i]
        decisive = _same(episode.get('episode_id'), decisive_id)
        text = ('* ' if decisive else '') + _episode_text(episode)
        lines.append(NumberLine('episodes', '', text, mark=MARK_PASS if episode_accepted(episode) else MARK_NONE))
        if decisive:
            shown |= {ep for column, ep, _ in _EPISODE_FIELDS if column in episode} | {ep for column, ep in _EPISODE_IDENTITY}
    hidden = len(runs) - len(listed)
    if hidden:
        lines.append(NumberLine('episodes', '', f'… {hidden} more rejected runs (episodes.csv)', mark=MARK_NONE))
    if decisive_id is not None and not missing(decisive_id) and shown:
        lines.append(NumberLine('episodes', '', '* the decisive episode (its ep_* columns follow)'))
    return lines, shown


def _same(a: Any, b: Any) -> bool:
    if missing(a) or missing(b):
        return False
    try:
        return int(float(a)) == int(float(b))
    except (TypeError, ValueError):
        return str(a) == str(b)


# constants.csv: planet columns of the two tables, then every other column of a row as 'name value'
_PLANET_TABLES = (
    (('n_deg_yr', 'n °/yr', 9), ('n_source', 'src', 5), ('mu', 'mu', 10), ('a_mean_au', 'a au', 10), ('e_mean', 'e', 9)),
    (
        ('g_arcsec_yr', 'g ″/yr', 8),
        ('varpi_dot_meas_arcsec_yr', "ϖ' ″/yr", 9),
        ('s_arcsec_yr', 's ″/yr', 8),
        ('Omega_dot_meas_arcsec_yr', "Ω' ″/yr", 9),
    ),
)
_SOURCE_SHORT = {'measured': 'meas', 'catalogue': 'cat', 'catalog': 'cat', 'const': 'cat'}


def _cell_text(column: str, value: Any) -> str:
    if column == 'n_source' and not missing(value):
        return _SOURCE_SHORT.get(str(value), str(value)[:4])
    return format_cell(column, value)


def _constant_lines(constants: Sequence[Mapping[str, Any]]) -> List[NumberLine]:
    """constants.csv: the planets as two aligned tables (mean motions and masses; secular frequencies),
    then one line per term (great inequality, near-commensurabilities) with all its numbers."""
    planets = [row for row in constants if any(column in row for table in _PLANET_TABLES for column, *_ in table)]
    terms = [row for row in constants if row not in planets]
    lines = []
    for index, table in enumerate(_PLANET_TABLES):
        present = [(column, title, width) for column, title, width in table if any(column in row for row in planets)]
        if not present:
            continue
        group = 'planets: mean motion, mass, a, e' if index == 0 else 'planets: secular frequencies'
        header = f"{'':<8}" + ''.join(f'{title:>{width}}' for _, title, width in present)
        lines.append(NumberLine('constants', '', header, group=group))
        for row in planets:
            name = str(row.get('planet', ''))[:7]
            cells = ''.join(f'{_cell_text(column, row.get(column)):>{width}}' for column, _, width in present)
            lines.append(NumberLine('constants', '', f'{name:<8}{cells}', group=group))
    known = {column for table in _PLANET_TABLES for column, *_ in table} | {'planet'}
    for row in planets:  # a planet column of no table is still printed
        extra = [
            f'{column} {format_cell(column, value)}{numbers.unit_space(column)}' for column, value in row.items() if column not in known
        ]
        if extra:
            lines.append(NumberLine('constants', '', f"{row.get('planet')}: " + ', '.join(extra), group='planets: other'))
    for row in terms:
        parts = [
            f'{label_of(column)} {format_cell(column, value)}{numbers.unit_space(column) if not missing(value) else ""}'
            for column, value in row.items()
            if column != 'planet'
        ]
        lines.append(NumberLine('constants', '', f"{row.get('planet')}: " + ', '.join(parts), group='planetary terms'))
    return lines


# --------------------------------------------------------------------------------------------- layout (combined_numbers)


def draw_text(fig, blocks: Dict[str, TextBlock], layout: VerdictFigureLayout) -> Dict[str, int]:
    """The three text columns, blocks in TEXT_BLOCKS order (the tests plan, place and list them through
    `combined_numbers` with TEXT_BLOCKS); returns `place_blocks` statistics."""
    return numbers.draw_text(fig, blocks, layout, TEXT_BLOCKS)


# --------------------------------------------------------------------------------------------- explained column (A′)

# The numbers of the explained column, most important first (the author's N3); a group with no value is left out.
# Everything else stays in summary.csv / mmr_numbers.csv and in the 'all' form of the column.
EXPLAINED_GROUPS = (
    (
        'LIBRATION (deciding episode)',
        (
            'ep_centre_mod_2pi_deg',
            'ep_n_tp',
            'ep_period_med_yr',
            'ep_half_amp_med_deg',
            'ep_half_amp_max_deg',
            'ep_centre_shift_over_amp',
            'ep_coverage_start_gap_P',
            'ep_coverage_end_gap_P',
            'geo_n_tp',
            'geo_period_cycle_yr',
            'geo_centre_window_yr',
            'geo_centre_max_pass_rev',
            'geo_drift_fraction',
            'geo_final_circulation_P',
            'geo_n_slips',
            'geo_n_revolutions',
            'geo_pattern_b_detected',
            'geo_n_episodes_before_jump',
        ),
    ),
    (
        'RESPONSE OF a',
        (
            'rsp_identity_slope_ratio',
            'rsp_identity_corr',
            'rsp_b1_ratio',
            'rsp_force_law',
            'rsp_force_window_yr',
            'rsp_force_n_windows',
            'rsp_force_R2_h1',
            'rsp_force_R2_h2',
            'rsp_force_phi0_deg',
            'rsp_phi0_max_dev_deg',
            'rsp_phi0_tol_deg',
            'rsp_coorb_regime',
            'rsp_coorb_K_over_K_theory',
            'rsp_coorb_energy_R2',
            'rsp_omega0_ratio',
            'rsp_b5_ratio',
        ),
    ),
    (
        'PENDULUM, ENERGY, ZONE',
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
        'SUBRESONANCES, OWNER',
        (
            'sub_triggers',
            'sub_n_scanned',
            'sub_best',
            'sub_best_phi0_coherence',
            'sub_accepted',
            'sub_theory_measured',
            'sub_flags',
            'own_name',
            'own_n_candidates',
            'own_flags',
        ),
    ),
    (
        'RESONANCE, RATES',
        (
            'ctx_order',
            'ctx_a_res_nominal_au',
            'ctx_a_star_au',
            'ctx_a_mean_au',
            'ctx_delta_a_au',
            'ctx_a_from_mean_longitude_rate_au',
            'ctx_c_rad_yr_au',
            'ctx_rate_expected_deg_yr',
            'ctx_rate_measured_deg_yr',
        ),
    ),
    (
        'STEP, ALIAS, FILTER',
        (
            'smp_step_yr',
            'smp_n_rows',
            'smp_verdict',
            'smp_step_rotation_deg_max',
            'smp_unwrap_resid_p99_deg',
            'smp_far_ratio',
            'flt_cutoff_yr',
            'flt_cutoff_binding',
            'flt_gain_at_pcyc',
            'flt_filter_too_slow',
            'flt_sensitive_quantities',
            'flt_withdrawn_before_tree',
            'rsp_response_cutoff_yr',
        ),
    ),
    (
        'STRENGTH',
        # CS11: the numbers are in the physics block ('Is σ the own pendulum'), not repeated here
        ('str_method', 'str_n_consistent', 'str_n_evaluations', 'str_planet_elements', 'str_not_measured_reason'),
    ),
    ('KINEMATICS', ('kin_corr_rate', 'kin_e_free', 'kin_e_forced', 'kin_modulation_period_kyr', 'kin_two_omega_period_yr')),
    ('ORBIT', ORBIT_COLUMNS[:-1]),
    ('RECORD', ('flags', 'status_code', 'rec_rule', 'settings_hash', 'confidence_cells_sha')),  # caps, thin: in CONFIDENCE
)
EXPLAINED_PREFIXES = ('ep_', 'geo_', 'rsp_', 'sep_', 'sub_', 'own_', 'ctx_', 'smp_', 'flt_', 'str_', 'kin_', 'rec_')
REJECTED_ROWS_MAX = 4  # rejected runs listed one by one in the explained column (the longest)
HASH_CHARS = 10
# constants.csv rows of planetary terms -> their name on the figure
TERM_NAMES = {
    'great_inequality': 'great ineq. 2J−5S',
    'venus_earth_8_13': 'V–E 8:13',
    'saturn_uranus_1_3': 'S–U 1:3',
    'jupiter_uranus_1_7': 'J–U 1:7',
    'uranus_neptune_1_2': 'U–N 1:2',
}
SECULAR_MODES_SHOWN = (('g', 'Jupiter'), ('g', 'Saturn'), ('s', 'Saturn'))  # g5, g6, s6: always listed
ARCSEC_PER_TURN = 1296000.0


def _period_from_rate(rate_arcsec_yr: Any) -> float:
    try:
        rate = abs(float(rate_arcsec_yr))
    except (TypeError, ValueError):
        return float('nan')
    return ARCSEC_PER_TURN / rate if rate > 0 and math.isfinite(rate) else float('nan')


def _mode_label(kind: str, planet: str) -> str:
    from resonances.resonance.analysis.mmr.models import PLANET_NAMES_ORDER

    return f'{kind}{PLANET_NAMES_ORDER.index(planet) + 1}'


def _synodic_planets(row: Mapping[str, Any]) -> Dict[str, float]:
    """{planet: synodic period} of the angle's planets, from ctx_P_syn_yr ('J:10.88;S:6.999')."""
    from resonances.resonance.analysis.mmr.models import PLANET_LETTERS

    synodic = {}
    for item in str(row.get('ctx_P_syn_yr') or '').split(';'):
        letter, _, value = item.partition(':')
        if letter.strip() in PLANET_LETTERS and value:
            synodic[PLANET_LETTERS[letter.strip()]] = _float(value)
    return synodic


def _term_note(item: Mapping[str, Any]) -> str:
    measured = item.get('measured')
    if measured is None:
        return ''
    return 'measured' if str(measured).lower() in ('true', '1') else 'table'


def period_lines(row: Mapping[str, Any], constants: Sequence[Mapping[str, Any]]) -> List[NumberLine]:
    """CHARACTERISTIC PERIODS: what a period seen on the figure may be (the author's 16b (d)): the synodic periods of
    the angle's planets, the body's orbit and perihelion, ϖ − ϖ_P of the angle's planets (the splitting of the
    multiplet, the wobble of the centre), the planetary near-commensurabilities (the great inequality ~ 900 yr,
    U–N 1:2 ~ 4.2 kyr, ...) and the secular modes g5, g6, s6 (and those of the angle's planets)."""
    planets = {str(item.get('planet')): item for item in constants if 'g_arcsec_yr' in item}
    synodic = _synodic_planets(row)
    varpi_rate = _float(row.get('ctx_varpi_dot_arcsec_yr'))
    periods = [(f'synodic with {planet[:3]}', period, '') for planet, period in synodic.items()]
    periods += [('orbit P_orb', _float(row.get('ctx_P_orb_yr')), ''), ('ϖ of the body', _period_from_rate(varpi_rate), '')]
    for planet in synodic:
        g = _float(planets.get(planet, {}).get('g_arcsec_yr'))
        periods.append((f'ϖ − ϖ_{planet[:3]} ({_mode_label("g", planet)})', _period_from_rate(varpi_rate - g), ''))
    for item in constants:
        name = str(item.get('planet'))
        if name not in planets and 'period_yr' in item:
            periods.append((TERM_NAMES.get(name, name.replace('_', ' ')), _float(item.get('period_yr')), _term_note(item)))
    modes = list(SECULAR_MODES_SHOWN) + [(kind, planet) for planet in synodic for kind in ('g', 's')]
    for kind, planet in dict.fromkeys(modes):
        rate = _float(planets.get(planet, {}).get(f'{kind}_arcsec_yr'))
        periods.append((f'{_mode_label(kind, planet)} ({planet[:3]})', _period_from_rate(rate), ''))
    group = 'CHARACTERISTIC PERIODS'
    return [
        NumberLine('numbers', label, years(period) + (f'  {note}' if note else ''), group=group)
        for label, period, note in periods
        if math.isfinite(period)
    ]


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def explained_episode_lines(episodes: Sequence[Mapping[str, Any]], decisive_id: Any = None) -> List[NumberLine]:
    """EPISODES: every accepted run (* = the deciding one) and the longest rejected runs with their reason."""
    accepted = [episode for episode in episodes if episode_accepted(episode)]
    rejected = [episode for episode in episodes if not episode_accepted(episode)]
    waves = sum(1 for episode in accepted if _forced_wave(episode))
    counted = f'{len(accepted)} accepted' + (f' ({waves} forced {"wave" if waves == 1 else "waves"})' if waves else '')
    group = f'EPISODES ({counted}, {len(rejected)} rejected runs)'
    lines = []
    for episode in accepted:
        mark = '*' if _same(episode.get('episode_id'), decisive_id) else ' '
        wave = _forced_wave(episode)
        text = (
            f"{mark}#{format_cell('episode_id', episode.get('episode_id'))} {years(episode.get('t_start_yr'))}–"
            f"{years(episode.get('t_end_yr'))} {fmt(episode.get('n_tp'))} TP P {years(episode.get('period_med_yr'))} "
            f"A {fmt(episode.get('half_amp_med_deg'))}/{fmt(episode.get('half_amp_max_deg'))}°" + (': forced wave' if wave else '')
        )
        lines.append(NumberLine('numbers', '', text, group=group, style='grey' if wave else 'bold'))
    rejected.sort(key=lambda episode: -_float(episode.get('n_tp')))
    for episode in rejected[:REJECTED_ROWS_MAX]:
        reason = str(episode.get('reject_reason') or '').replace('_', ' ')
        text = f"✗ {years(episode.get('t_start_yr'))}–{years(episode.get('t_end_yr'))} {fmt(episode.get('n_tp'))} TP: {reason}"
        lines.append(NumberLine('numbers', '', text, group=group, style='grey'))
    if len(rejected) > REJECTED_ROWS_MAX:
        lines.append(
            NumberLine('numbers', '', f'… {len(rejected) - REJECTED_ROWS_MAX} more rejected (episodes.csv)', group=group, style='grey')
        )
    if not episodes:
        lines.append(NumberLine('numbers', '', 'no runs of turning points', group=group, style='grey'))
    return lines


SUB_TABLE_ROWS = 3  # members of the ranked list with columns of their own (`subresonances.TABLE_ROWS`)
SUB_KEY_CHARS = 14  # the key column of the table; a longer key takes a line of its own above its numbers
SUB_CELL_CHARS = 5  # one number of the table, right-aligned after a space (the line keeps its 38 characters)


def _sub_cell(value: Any) -> str:
    """Two significant digits in at most SUB_CELL_CHARS characters: the sentence form (`fmt`), or a short exponent when
    that is longer (0.0019 -> 2e-3, -4 020 658 -> -4e6)."""
    if missing(value):
        return '—'
    text = fmt(value, 2)
    if len(text) <= SUB_CELL_CHARS:
        return text
    mantissa, exponent = f'{float(value):.0e}'.split('e')
    return f'{mantissa}e{int(exponent)}'


def _table_lines(key: str, cells: Sequence[Any], group: str, column: str = '', style: str = '') -> List[NumberLine]:
    """One row of the subresonance table (a key longer than SUB_KEY_CHARS on a line of its own); every cell after a
    space, so two numbers never run together."""
    numbers_text = ''.join(f' {cell:>{SUB_CELL_CHARS}}' for cell in cells)
    if len(key) < SUB_KEY_CHARS:
        return [NumberLine('numbers', '', f'{key:<{SUB_KEY_CHARS}}{numbers_text}', column=column, group=group, style=style)]
    return [
        NumberLine('numbers', '', key, column=column, group=group, style=style),
        NumberLine('numbers', '', f'{"":<{SUB_KEY_CHARS}}{numbers_text}', group=group, style=style),
    ]


def ranked_members_lines(row: Mapping[str, Any], group: str) -> List[NumberLine]:
    """S2 (W20 CS9, drawn in CS11): the multiplet's best members as a table, W/W0 the range of the member's slow
    centre against the catalogue angle's own W0, E the share of the catalogue centre's motion it explains, N its own
    slow cycles in the scan window, A/W σ0's median full swing over W; then the catalogue angle itself (W0 in degrees,
    A/W0) and, for a two-body row with a moving-well track, the minimum of the averaged potential (its W against W0).
    N < 1 is said in words (only a rate matched), and the d'Alembert caveat when the factors are unreliable."""
    members = [i for i in range(1, SUB_TABLE_ROWS + 1) if str(row.get(f'sub_rank{i}_key') or '')]
    if not members:
        return []
    lines = _table_lines('member', ('W/W0', 'E', 'N', 'A/W'), group)
    short = False
    for i in members:
        cells = [row.get(f'sub_rank{i}_{name}') for name in ('W_ratio', 'E', 'N', 'AW')]
        short = short or (not missing(cells[2]) and float(cells[2]) < 1)
        lines += _table_lines(str(row[f'sub_rank{i}_key']), [_sub_cell(c) for c in cells], group, f'sub_rank{i}_key')
    w0 = row.get('sub_cat_W_deg')
    label = f'σ W0 {_sub_cell(w0)}°' if not missing(w0) else 'σ (catalogue)'
    lines += _table_lines(label, ('1', '0', '—', _sub_cell(row.get('sub_cat_AW'))), group, 'sub_cat_W_deg')
    w = row.get('sub_pot_W_deg')
    if not missing(w):
        ratio = float(w) / float(w0) if not missing(w0) and float(w0) > 0 else float('nan')
        lines += _table_lines('pot. minimum', (_sub_cell(ratio), _sub_cell(row.get('sub_pot_E')), '—', '—'), group, 'sub_pot_W_deg')
    if short:
        lines.append(
            NumberLine('numbers', '', 'N < 1: less than one slow cycle in the window, only a rate matched', group=group, style='grey')
        )
    caveat = str(row.get('sub_strength_caveat') or '')
    if caveat:
        lines.append(NumberLine('numbers', '', f"d'Alembert factors unreliable: {caveat.replace(';', ', ')}", group=group, style='grey'))
    return lines


def _without(groups, columns) -> tuple:
    """EXPLAINED_GROUPS without the columns the physics block prints (no number twice)."""
    return tuple((title, tuple(c for c in group if c not in columns)) for title, group in groups)


def short_hashes(row: Dict[str, Any]) -> Dict[str, Any]:
    """`row` with its settings_hash and confidence_cells_sha cut to HASH_CHARS (the explained columns print them so)."""
    for column in ('settings_hash', 'confidence_cells_sha'):
        if not missing(row.get(column)):
            row[column] = str(row[column])[:HASH_CHARS]
    return row


def config_settings(settings=None):
    """`settings`, else the AnalysisSettings of the configuration (an error when it cannot be read)."""
    if settings is not None:
        return settings
    from resonances.resonance.analysis.settings import settings_from_config

    return settings_from_config()


def explained_lines(
    summary: Mapping[str, Any],
    numbers_row: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    constants: Sequence[Mapping[str, Any]],
    text,
) -> List[NumberLine]:
    """The numbers of the explained column: episodes first, then EXPLAINED_GROUPS with the characteristic periods
    after the resonance, bold = compared by the decision, grey = of a test skipped by design (`text`: VerdictText)."""
    row = short_hashes({**numbers_row, **summary})
    lines = explained_episode_lines(episodes, row.get('ep_episode_id'))
    shown = set(physics_columns(getattr(text, 'physics', ())))
    for title, columns in _without(EXPLAINED_GROUPS, shown):
        lines += numbers.group_lines(row, ((title, columns),), EXPLAINED_PREFIXES, text.bold_columns, text.grey_columns)
        if title == 'SUBRESONANCES, OWNER':
            lines += ranked_members_lines(row, title)
        if title == 'RESONANCE, RATES':
            lines += period_lines(row, constants)
    return lines


def draw_explained(fig, analysis, rows, layout: VerdictFigureLayout, settings=None) -> Dict[str, Any]:
    """The explained right column of an MMR figure; `rows` = (summary, numbers, episodes, constants)."""
    summary, numbers_row, episodes, constants = rows
    text = explain_mmr(analysis, config_settings(settings), {**numbers_row, **summary}, episodes)
    stats = numbers.draw_explained(fig, text, explained_lines(summary, numbers_row, episodes, constants, text), layout)
    stats['text'] = text
    return stats
