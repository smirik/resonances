"""The V1 combined figure after the W2 'figure' package: text order and format, the centre panels (j), (k).

Synthetic analyses built by hand (no integration). The truths are independent of the code:
- the members of a multiplet follow from d'Alembert (same lambda part, eccentricity powers of one sign summing
  to the order): 3J-2+0-1 has the single other member 3J-2-1+0, sigma1 - sigma0 = varpi - varpi_J;
- a Hilda-like record is built from a member that librates about pi while varpi - varpi_J circulates, so the
  catalogue angle's centre circulates and the member's line in (k) must be flat;
- printed numbers are checked against hand-written strings (4 significant digits, whole numbers from 1e4,
  1.234e-05 below 1e-3).
"""

import typing
from dataclasses import MISSING, fields

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from resonances.plotting import combined_numbers as cn  # noqa: E402
from resonances.plotting import mmr_combined as mc  # noqa: E402
from resonances.plotting import mmr_verdict_panel as panel  # noqa: E402
from resonances.plotting.style import SUBRESONANCE, OWNER  # noqa: E402
from resonances.resonance.analysis import geometry as geo  # noqa: E402
from resonances.resonance.analysis import pendulum as pend  # noqa: E402
from resonances.resonance.analysis.filtering import lowpass  # noqa: E402
from resonances.resonance.analysis.ledger import Ledger  # noqa: E402
from resonances.resonance.analysis.mmr import models as m  # noqa: E402
from resonances.resonance.analysis.verdict import (  # noqa: E402
    AngleStatus,
    CheckValue,
    Confidence,
    LibrationOrigin,
    Reason,
    ResonanceType,
    Status,
    Verdict,
)

NAN = float('nan')
T = np.arange(0.0, 20001.0)  # 20 kyr, step 1 yr
P_LIB = 270.0  # libration period [yr]
P_BETA = 2600.0  # period of varpi - varpi_J [yr] (Hilda: 2.6 kyr)
W_C = 3 * P_LIB  # the centre window, 3 cycles
A_RES = 3.97
N_BODY = 2 * np.pi * A_RES**-1.5
N_JUPITER = 2 * np.pi / 11.86
HILDA = m.ResonanceSpec('3J-2+0-1', m.ResonanceKind.TWO_BODY, ('Jupiter',), (3,), -2, -1, (0,), nominal_axis_au=A_RES)


def make(cls, **given):
    """A dataclass with every field not given set to its 'not measured' value (nan, empty, None)."""
    hints = typing.get_type_hints(cls, vars(m) | vars(geo) | vars(pend) | {'np': np})
    values = {}
    for f in fields(cls):
        if f.name in given or f.default is not MISSING or f.default_factory is not MISSING:
            continue
        values[f.name] = _empty(hints[f.name])
    values.update(given)
    return cls(**values)


def _empty(hint):
    origin = typing.get_origin(hint)
    simple = {float: NAN, bool: False, int: 0, str: '', np.ndarray: np.array([])}
    if hint in simple:
        return simple[hint]
    if origin in (tuple, list, dict):
        return origin()
    if hint is CheckValue:
        return CheckValue.NOT_MEASURED
    return None


def hilda_like():
    """sigma1 = 3 lJ - 2 l - varpi_J librates about pi (P 270 yr, 23 deg); beta = varpi - varpi_J circulates
    (2.6 kyr), so sigma0 = sigma1 - beta has a centre that circulates with beta."""
    varpi_j = np.full_like(T, 0.25)
    varpi = 1.0 - 2 * np.pi * T / P_BETA
    sigma1 = np.pi + 0.4 * np.sin(2 * np.pi * T / P_LIB)
    sigma0 = sigma1 - (varpi - varpi_j)
    lam_j = N_JUPITER * T
    lam = (3 * lam_j - varpi - sigma0) / 2  # sigma0 = 3 lJ - 2 l - varpi exactly
    planet = m.PlanetSeries('Jupiter', T, np.full_like(T, 5.2), np.full_like(T, 0.048), np.zeros_like(T), np.zeros_like(T), varpi_j, lam_j)
    regime = m.Regime(step_yr=1.0, span_yr=2e4, n_rows=len(T), integrator='SABA(10,6,4)', dt=1.0, epoch_jd=2461319.5)
    a = A_RES + 0.01 * np.cos(2 * np.pi * T / P_LIB)
    series = m.MMRSeries(
        t_yr=T,
        a=a,
        e=0.19 + 0.05 * np.cos(2 * np.pi * T / P_BETA),
        inc=np.radians(8 + np.sin(2 * np.pi * T / 4000)),
        Omega=np.zeros_like(T),
        omega=np.mod(varpi, 2 * np.pi),
        varpi=np.mod(varpi, 2 * np.pi),
        angle_wrapped=np.mod(sigma0, 2 * np.pi),
        resonance=HILDA,
        regime=regime,
        planets={'Jupiter': planet},
        mean_longitude=np.mod(lam, 2 * np.pi),
        mean_anomaly=np.mod(lam - varpi, 2 * np.pi),
        name='hilda_like',
    )
    return series, sigma0, sigma1


def analysis_of(series, sigma0):
    rate = np.gradient(sigma0, T)
    window = (W_C, T[-1] - W_C)
    prep = make(
        m.Preprocessed,
        cutoff=m.Cutoff(35.4, 'synodic', 35.4, 20.0, 35.4),
        t_yr=T,
        angle_unwrapped=sigma0,
        angle_clean=sigma0,
        angle_f=sigma0,
        rate_f=rate,
        angle_f_half=sigma0,
        angle_f_double=sigma0,
        a_f=series.a,
        adot_f=np.gradient(series.a, T),
        e_f=series.e,
        inc_f=series.inc,
        window=window,
        ripple_rad=None,
    )
    c0 = lowpass(T, sigma0, W_C, 2)
    turns = geo.TurningPoints(np.array([], int), np.array([]), np.array([]), np.array([]), 0.2)
    centre = geo.CentreTrack(T, c0, W_C, turns, [], 7.0, 1000.0, 3600.0, -7.0, 40.0, 1.1, 1.2)
    tp_t = np.arange(P_LIB / 4, T[-1], P_LIB / 2)
    kinds = np.where(np.arange(len(tp_t)) % 2 == 0, 1, -1)
    tp = geo.TurningPoints(tp_t.astype(int), tp_t, np.interp(tp_t, T, sigma0), kinds, 0.05)
    geometry = make(geo.Geometry, window=window, turning_points=tp, centre=centre, runs=[], period_cycle_yr=P_LIB)
    cycles = [pend.CycleEnergy(t0, t0 + P_LIB, False, 0.2, NAN, P_LIB, 1.0) for t0 in np.arange(500.0, 19000.0, P_LIB)]
    ledger = Ledger(thin_band=0.1)
    ledger.compare('L1 TP', 147, '>=', 7, '', 'geo_n_tp')
    verdict = Verdict(
        ResonanceType.MMR,
        Status.UNCERTAIN,
        AngleStatus.CIRCULATION,
        confidence=Confidence.LOW,
        reason=Reason.CIRCULATION,
        libration_origin=LibrationOrigin.UNKNOWN,
        checks={'check_mmr_subresonances': CheckValue.NOT_APPLICABLE, 'check_mmr_owner': CheckValue.NOT_APPLICABLE},
        decision_path=ledger.entries,
    )
    context = make(
        m.Context,
        planet_mean_motion_rad_yr={'Jupiter': N_JUPITER},
        planet_mean_motion_source={'Jupiter': 'measured'},
        planet_mass={'Jupiter': 9.548e-4},
        planet_varpi_rate_rad_yr={},
        great_inequality=None,
        order=1,
        a_mean_au=A_RES,
        nominal_axis_au=A_RES,
        exact_axis_au=A_RES + 0.002,
        axis_from_mean_longitude_rate_au=A_RES - 0.003,
        identity_coefficient=0.6,
        rate_expected_rad_yr=1e-4,
        fair_step_fraction=1 / A_RES**1.5,
        crosses=(),
        synodic_periods_yr={'Jupiter': 23.6},
    )
    return make(
        m.MMRAnalysis,
        series_name='hilda_like',
        resonance=HILDA,
        context=context,
        sampling=make(m.Sampling, angle_unwrapped=sigma0, residual_rad=np.zeros(len(T) - 1)),
        prep=prep,
        slow=None,
        geometry=geometry,
        energy=make(pend.EnergyAnalysis, potential_e=None, potential_f=None, cycles=cycles, zone=None),
        response=make(m.Response, windows=[], multiplet=None),
        kinematics=None,
        subresonances=None,
        owner=None,
        strength=None,
        baseline=None,
        episode_physics=[],
        verdict=verdict,
        settings_hash='0123456789ab',
        confidence_cells_sha='',
    )


def rows_of(analysis):
    summary = {**analysis.verdict.to_flat_dict(), 'name': analysis.series_name, 'resonance': analysis.resonance.key, 'geo_n_tp': 147}
    numbers = {'geo_centre_window_yr': W_C, 'geo_period_cycle_yr': P_LIB, 'smp_step_yr': 1.0}
    return summary, numbers, [], []


def _ptp_deg(values):
    values = np.asarray(values, dtype=float)
    return float(np.degrees(np.ptp(np.unwrap(values[np.isfinite(values)]))))


# --------------------------------------------------------------------------------------------- text format


class TestCells:
    @pytest.mark.parametrize(
        'column, value, text',
        [
            ('flt_window_t1_yr', 99964.5748, '99965'),  # >= 1e4: a whole number, not 9.996e+04
            ('smp_far_ratio', 1.8415e-4, '1.842e-04'),  # < 1e-3: 1.234e-05 style
            ('smp_jitter', 0.0057556, '0.005756'),  # 1e-3 .. 1e4: 4 significant digits, fixed
            ('mrg_L2_value', 30.145940944836294, '30.15'),
            ('geo_n_tp', 739, '739'),
            ('thin_margins', 'L4;L5;R0;F', 'L4; L5; R0; F'),
            ('ctx_a_star_au', 3.9654029218, '3.965403'),  # semi-major axis: 7 digits
            ('geo_centre_tp_times_yr', ';'.join(str(t) for t in range(873, 873 + 76 * 100, 100)), '76 values: 873 … 8373'),
        ],
    )
    def test_format(self, column, value, text):
        assert panel.format_cell(column, value) == text

    @pytest.mark.parametrize(
        'column, unit, label',
        [
            ('smp_step_rotation_deg_min', '°', 'step_rotation_min'),
            ('smp_step_rotation_deg_max', '°', 'step_rotation_max'),
            ('geo_net_rev_raw', 'rev', 'net_rev_raw'),
            ('smp_revs_predictive', 'rev', 'revs_predictive'),
            ('geo_centre_max_pass_rev_lo', 'rev', 'centre_max_pass_rev_lo'),
            ('ep_coverage_start_gap_P', 'P', 'coverage_start_gap'),
        ],
    )
    def test_units_before_a_qualifier(self, column, unit, label):
        assert panel.unit_of(column) == unit
        assert panel.label_of(column) == label


class TestDuplicates:
    def test_episode_copy_of_a_record_number_is_printed_once_where_the_record_prints_it(self):
        numbers = {'rsp_force_R2_h1': 0.70214, 'ep_force_R2_h1': 0.70214, 'flt_window_t0_yr': 35.4, 'geo_window_t0_yr': 35.4}
        blocks = panel.text_blocks({}, numbers, [])
        printed = [line for line in cn.printed_lines(blocks, panel.TEXT_BLOCKS) if line.column]
        force = [line for line in printed if line.label == 'force_R2_h1']
        assert len(force) == 1 and force[0].column == 'rsp_force_R2_h1' and force[0].also == ('ep_force_R2_h1',)
        window = [line for line in printed if line.label == 'window_t0']
        assert len(window) == 1 and window[0].also == ('geo_window_t0_yr',)
        # every column still maps to a printed line with its value
        columns = cn.printed_columns(blocks, panel.TEXT_BLOCKS)
        assert columns['ep_force_R2_h1'].value == '0.7021' and columns['geo_window_t0_yr'].value == '35.4'

    def test_different_quantities_are_kept(self):
        numbers = {
            'rsp_force_R2_h1': 0.70214,
            'ep_force_R2_h1': 0.5,  # another value: not a duplicate
            'exit_suspected': False,  # same label 'suspected', different quantities
            'entry_suspected': False,
            'rsp_flags': '',  # empty cells are never merged
            'sep_flags': '',
        }
        blocks = panel.text_blocks({}, numbers, [])
        columns = {line.column for line in cn.printed_lines(blocks, panel.TEXT_BLOCKS) if line.column}
        assert set(numbers) <= columns


class TestOrderAndPlacement:
    def placement(self, numbers=None):
        analysis, _ = Case.hilda()
        summary, base, episodes, constants = rows_of(analysis)
        row = {**base, **(numbers or {})}
        blocks = panel.text_blocks(summary, row, analysis.verdict.decision_path, episodes, constants)
        return blocks, cn.plan_columns(blocks, mc.LAYOUT_V1, panel.TEXT_BLOCKS)

    def test_block_order(self):
        _, placement = self.placement()
        titles = [row.text for column in placement.columns for row in column if row.kind == 'title' and '(cont.)' not in row.text]
        expected = [panel.BLOCKS[name].title for name in panel.TEXT_BLOCKS]
        assert titles == expected
        assert expected[:6] == ['VERDICT', 'DECISION PATH (ledger)', 'CHECKS check_mmr_*', 'MARGINS', 'SUBRESONANCES', 'OWNER']
        assert expected[-1] == 'ORBIT: e, i, q, Q'

    def test_a_block_cut_by_a_column_continues_under_its_title(self):
        numbers = {f'geo_extra_{i:03d}_deg': i + 0.5 for i in range(150)}
        blocks, placement = self.placement(numbers)
        firsts = [column[0] for column in placement.columns[1:] if column]
        assert any(row.kind == 'title' and row.text.endswith(' (cont.)') for row in firsts)
        cells = [cell.label for column in placement.columns for row in column for cell in row.cells]
        assert all(cells.count(f'extra_{i:03d}') == 1 for i in range(150))  # nothing cut, nothing twice

    def test_group_headers(self):
        numbers = {'geo_n_tp': 147, 'geo_centre_window_yr': W_C, 'exit_time_yr': 100.0, 'jump_tau_yr': 70.0}
        blocks, placement = self.placement(numbers)
        groups = [row.text for column in placement.columns for row in column if row.kind == 'group' and row.block == 'GEOMETRY']
        assert groups == ['turning points', 'centre', 'exit and entry', 'jumps of a']


class TestTables:
    def test_long_episode_list_is_counted(self):
        episodes = [
            {
                'episode_id': i,
                'accepted': i == 5,
                'subtype': 'stationary',
                'reject_reason': 'too_few_tp',
                't_start_yr': 100.0 * i,
                't_end_yr': 100.0 * i + 50,
                'n_tp': 2,
            }
            for i in range(20)
        ]
        blocks = panel.text_blocks({}, {'ep_episode_id': 5, 'ep_n_tp': 2}, [], episodes)
        texts = [line.value for line in blocks['episodes'].lines]
        assert texts[0].startswith('#0 ') and texts[1].startswith('* #5 ') and texts[2].startswith('#19 ')
        assert '… 17 more rejected runs (episodes.csv)' in texts
        # the decisive episode's ep_n_tp is the table's value: not printed again
        assert 'ep_n_tp' not in {line.column for line in blocks['episodes'].lines}

    def test_every_constant_is_printed(self):
        constants = [
            {
                'planet': 'Jupiter',
                'n_deg_yr': 30.3496,
                'n_source': 'measured',
                'mu': 9.548e-4,
                'a_mean_au': 5.2026,
                'e_mean': 0.0484,
                'g_arcsec_yr': 4.257,
                's_arcsec_yr': 0.0,
                'varpi_dot_meas_arcsec_yr': 4.2,
                'Omega_dot_meas_arcsec_yr': -26.3,
            },
            {
                'planet': 'great_inequality',
                'period_yr': 938.0,
                'dlambda_jupiter_deg': 0.3258,
                'dlambda_saturn_deg': 0.8017,
                'measured': True,
            },
        ]
        blocks = panel.text_blocks({}, {}, [], (), constants)
        text = '\n'.join(line.value for line in blocks['constants'].lines)
        for value in ('30.35', 'meas', '9.548e-04', '5.2026', '0.0484', '4.257', '4.2', '-26.3', '938', '0.3258', '0.8017'):
            assert value in text, value


# --------------------------------------------------------------------------------------------- members


class TestMembers:
    @pytest.mark.parametrize(
        'key, members',
        [
            ('3J-2+0-1', {'3J-2-1+0'}),
            ('5J-2+0-3', {'5J-2-1-2', '5J-2-2-1', '5J-2-3+0'}),
            ('3J-1S-1+0+0-1', {'3J-1S-1-1+0+0', '3J-1S-1+0-1+0'}),
            ('1J-3r0', set()),
        ],
    )
    def test_planar_members(self, key, members):
        assert {member.key for member in mc.planar_members(m.ResonanceSpec.from_key(key))} == members

    def test_order_seven_has_seven_other_members(self):
        members = mc.planar_members(m.ResonanceSpec.from_key('9M-16+0+7'))
        assert len(members) == 7 and '9M-16+7+0' in {member.key for member in members}

    def test_offset_is_varpi_minus_varpi_j(self):
        # sigma1 - sigma0 = (3 lJ - 2 l - varpi_J) - (3 lJ - 2 l - varpi) = varpi - varpi_J
        assert mc.varpi_offset(HILDA, m.ResonanceSpec.from_key('3J-2-1+0')) == (1, (-1,))
        assert mc.varpi_offset(HILDA, m.ResonanceSpec.from_key('4J-2+0-2')) is None  # another lambda part

    def test_spec_angle_rebuilds_the_saved_angle(self):
        series, sigma0, _ = hilda_like()
        angle = mc.spec_angle(HILDA, series)
        assert np.max(np.abs(np.angle(np.exp(1j * (angle - sigma0))))) < 1e-9

    def test_the_librating_member_is_flat_and_sigma0_is_not(self):
        series, sigma0, sigma1 = hilda_like()
        analysis = analysis_of(series, sigma0)
        lines = {line.key: line for line in mc.centre_lines(analysis, series, 2)}
        inside = (T >= W_C) & (T <= T[-1] - W_C)
        assert set(lines) == {'3J-2+0-1', '3J-2-1+0'}
        assert _ptp_deg(lines['3J-2+0-1'].centre_rad[inside]) > 360 * (T[-1] - 2 * W_C) / P_BETA * 0.9
        assert _ptp_deg(lines['3J-2-1+0'].centre_rad[inside]) < 3.0
        assert abs(np.angle(np.exp(1j * (np.median(lines['3J-2-1+0'].centre_rad[inside]) - np.pi)))) < 0.02

    def test_accepted_member_and_owner_carry_their_angles(self):
        series, sigma0, sigma1 = hilda_like()
        analysis = analysis_of(series, sigma0)
        member = make(m.SubresonanceCandidate, spec=m.ResonanceSpec.from_key('3J-2-1+0'), j=1, accepted=True, rank=0)
        analysis.subresonances = make(m.SubresonanceResult, accepted=member, candidates=[member], check=CheckValue.FAIL)
        owner = make(m.OwnerCandidate, spec=HILDA, accepted=True, r2_median=0.9, strength_ratio=1.0)
        analysis.owner = make(m.OwnerResult, accepted=owner, check=CheckValue.FAIL)
        lines = {line.role: line for line in mc.centre_lines(analysis, series, 2)}
        assert set(lines) == {'sigma0', 'subresonance', 'owner'}
        # the overlay of (b): sigma1 itself, librating about pi with 0.4 rad
        deviation = np.abs(np.angle(np.exp(1j * (lines['subresonance'].angle_wrapped - np.pi))))
        assert deviation.max() < 0.41
        # the owner here is the catalogue angle: its circular centre follows c0 inside the window
        inside = (T >= 2 * W_C) & (T <= T[-1] - 2 * W_C)
        difference = np.angle(np.exp(1j * (lines['owner'].centre_rad - lines['sigma0'].centre_rad)))[inside]
        assert np.max(np.abs(difference)) < 0.05


# --------------------------------------------------------------------------------------------- figure


class Case:
    @staticmethod
    def hilda():
        series, sigma0, _ = hilda_like()
        return analysis_of(series, sigma0), series


def _draw(analysis, series):
    fig = mc.draw_combined(analysis, series, rows=rows_of(analysis))
    fig.canvas.draw()
    return fig


def _texts(fig):
    return [text.get_text() for text in fig.findobj(matplotlib.text.Text)]


class TestFigure:
    def teardown_method(self):
        plt.close('all')

    def test_ten_panels_letters_and_the_same_rectangles(self):
        analysis, series = Case.hilda()
        fig = _draw(analysis, series)
        x, y, w, h = mc.LAYOUT_V1.panels
        width, height = mc.LAYOUT_V1.size
        n = mc.LAYOUT_V1.n_panels
        assert n == len(mc.PANELS) == 10
        gap = mc.LAYOUT_V1.panel_gap
        step = (h - (n - 1) * gap) / n
        expected = [(x / width, (y + h - (i + 1) * step - i * gap) / height, w / width, step / height) for i in range(n)]
        assert np.allclose([ax.get_position().bounds for ax in fig.axes[:n]], expected)
        texts = _texts(fig)
        for letter in 'abcdefghij':
            assert f'({letter})' in texts
        assert '(k) phase portrait, whole record' in texts and '(l) accepted episodes' in texts and '(m) FAIR' in texts
        analysis.prep = analysis.geometry = None  # an undefined record: same rectangles
        empty = _draw(analysis, series)
        assert np.allclose([ax.get_position().bounds for ax in empty.axes[:n]], expected)

    def test_centre_panels_carry_no_caption_only_the_members_keys(self):
        analysis, series = Case.hilda()
        fig = plt.figure(figsize=mc.LAYOUT_V1.size, dpi=mc.LAYOUT_V1.dpi)
        axes = mc.draw_time_panels(fig, analysis, series, mc.LAYOUT_V1)
        assert [text.get_text() for text in axes[8].texts] == ['(i)']  # W_c and the largest pass are in the column
        labels = [text.get_text() for text in axes[9].texts]
        assert labels[0] == '(j)' and '3J-2-1+0' in labels and not [text for text in labels if text.startswith('mode')]

    def test_accepted_subresonance_and_owner_are_overlaid_on_b(self):
        analysis, series = Case.hilda()
        member = make(m.SubresonanceCandidate, spec=m.ResonanceSpec.from_key('3J-2-1+0'), j=1, accepted=True)
        analysis.subresonances = make(m.SubresonanceResult, accepted=member, candidates=[member], check=CheckValue.FAIL)
        owner = make(m.OwnerCandidate, spec=m.ResonanceSpec.from_key('3J-2+0-1'), accepted=True, r2_median=0.9, strength_ratio=1.0)
        analysis.owner = make(m.OwnerResult, accepted=owner, check=CheckValue.FAIL)
        fig = _draw(analysis, series)
        wrapped = fig.axes[1]
        # both drawn filtered at the cutoff, as translucent broken lines under sigma_f (no raw points)
        overlays = {matplotlib.colors.to_hex(line.get_color()): line for line in wrapped.lines}
        assert SUBRESONANCE in overlays and OWNER in overlays
        assert overlays[OWNER].get_linestyle() == '--' and overlays[SUBRESONANCE].get_linestyle() == '-.'
        assert all(overlays[c].get_alpha() < 1 and overlays[c].get_zorder() < 5 for c in (OWNER, SUBRESONANCE))
        points = {matplotlib.colors.to_hex(c.get_facecolor()[0]) for c in wrapped.collections if len(c.get_facecolor())}
        assert SUBRESONANCE not in points and OWNER not in points
        labels = [(text.get_text(), matplotlib.colors.to_hex(text.get_color())) for text in fig.axes[9].texts]
        assert ('3J-2-1+0', SUBRESONANCE) in labels and ('3J-2+0-1', OWNER) in labels  # (j): keys in their colours

    def test_tick_labels_stay_inside_the_panels(self):
        analysis, series = Case.hilda()
        fig = _draw(analysis, series)
        margin = mc.LAYOUT_V1.tick_margin
        for ax in fig.axes[: mc.LAYOUT_V1.n_panels]:  # in axes fractions: (c) is symlog, (h) log
            low, high = sorted(ax.get_ylim())
            to_axes = ax.transData + ax.transAxes.inverted()
            fractions = [to_axes.transform((0.0, tick))[1] for tick in ax.get_yticks() if low <= tick <= high]
            assert all(margin * 0.999 <= f <= 1 - margin * 0.999 for f in fractions), ax.texts[0].get_text()

    def test_right_labels_and_legend_stay_in_their_boxes(self):
        analysis, series = Case.hilda()
        fig = _draw(analysis, series)
        renderer = fig.canvas.get_renderer()
        for ax in fig.axes[: mc.LAYOUT_V1.n_panels]:
            box = ax.get_window_extent(renderer)
            for text in ax.texts:
                if text.get_position()[0] > 1:  # labels at the right end of lines
                    extent = text.get_window_extent(renderer)
                    assert box.y0 - 1 <= extent.y0 and extent.y1 <= box.y1 + 1, text.get_text()
        x, y, w, h = mc.legend_rect(mc.LAYOUT_V1, third_slot_free=False)  # FAIR is drawn: the legend keeps its own box
        legend = max((text for text in fig.texts if text.get_text().startswith(mc.LEGEND[0][:12])), key=lambda t: len(t.get_text()))
        extent = legend.get_window_extent(renderer)
        assert extent.x1 <= (x + w) * fig.dpi + 1 and extent.y0 >= y * fig.dpi - 1

    def test_group_headers_are_bold(self):
        analysis, series = Case.hilda()
        summary, numbers, episodes, constants = rows_of(analysis)
        numbers = {**numbers, 'geo_n_tp': 147, 'jump_tau_yr': 70.0}
        # the group headers of the full dump (combined.numbers = 'all'; the explained column has its own: test_verdict_text)
        options = {'combined': {'numbers': 'all'}}
        fig = mc.draw_combined(analysis, series, rows=(summary, numbers, episodes, constants), options=options)
        headers = [text for text in fig.texts if text.get_text() in ('turning points', 'jumps of a')]
        assert len(headers) == 2 and all(text.get_fontweight() == 'bold' for text in headers)
