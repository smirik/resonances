"""The V1 combined figure (plotting.mmr_combined + mmr_verdict_panel) on synthetic analyses.

The analyses are built by hand (the analysis modules are not needed): a two-body libration with
FAIR, a three-body circulation without triggers and a record whose unwrap failed. The CSV rows
are passed explicitly, as after a restore from summary.csv."""

import math
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
from resonances.resonance.analysis import geometry as geo  # noqa: E402
from resonances.resonance.analysis import pendulum as pend  # noqa: E402
from resonances.resonance.analysis.ledger import Ledger  # noqa: E402
from resonances.resonance.analysis.mmr import models as m  # noqa: E402
from resonances.resonance.analysis.verdict import (  # noqa: E402
    AngleStatus,
    AngleSubtype,
    CheckValue,
    Confidence,
    LibrationOrigin,
    Reason,
    ResonanceType,
    Status,
    Verdict,
)

NAN = float('nan')


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
    if hint is float:
        return NAN
    if hint is bool:
        return False
    if hint is int:
        return 0
    if hint is str:
        return ''
    if hint is np.ndarray:
        return np.array([])
    if origin in (tuple,):
        return ()
    if origin in (list,):
        return []
    if origin in (dict,):
        return {}
    if hint is CheckValue:
        return CheckValue.NOT_MEASURED
    return None


T = np.arange(0.0, 10001.0)  # 10 kyr, step 1 yr
P_LIB = 500.0
A_RES = 3.97


def _libration_parts():
    phase = 2 * np.pi * T / P_LIB
    sigma = np.pi + 1.0 * np.sin(phase)
    rate = 1.0 * 2 * np.pi / P_LIB * np.cos(phase)
    a_f = A_RES + 0.01 * np.cos(phase)
    adot = -0.01 * 2 * np.pi / P_LIB * np.sin(phase)
    return sigma, rate, a_f, adot


def _series(spec, sigma, a, planets=True):
    n = 2 * np.pi * A_RES**-1.5
    n_jupiter = 2 * np.pi / 11.86
    varpi = 0.3 + 1e-4 * T
    longitude = n * T + 0.01 * np.sin(2 * np.pi * T / P_LIB)
    planet = m.PlanetSeries(
        'Jupiter', T, np.full_like(T, 5.2), np.full_like(T, 0.048), np.zeros_like(T), np.zeros_like(T), np.zeros_like(T), n_jupiter * T
    )
    regime = m.Regime(step_yr=1.0, span_yr=1e4, n_rows=len(T), integrator='SABA(10,6,4)', dt=1.0, epoch_jd=2461319.5)
    return m.MMRSeries(
        t_yr=T,
        a=a,
        e=0.15 + 0.02 * np.sin(2 * np.pi * T / 3000),
        inc=np.radians(8 + np.sin(2 * np.pi * T / 4000)),
        Omega=np.zeros_like(T),
        omega=varpi,
        varpi=varpi,
        angle_wrapped=np.mod(sigma, 2 * np.pi),
        resonance=spec,
        regime=regime,
        planets={'Jupiter': planet} if planets else {},
        mean_longitude=longitude,
        mean_anomaly=longitude - varpi,
        name='synthetic',
    )


def _context(**given):
    base = dict(
        planet_mean_motion_rad_yr={'Jupiter': 2 * np.pi / 11.86},
        planet_mean_motion_source={'Jupiter': 'measured'},
        planet_mass={'Jupiter': 9.548e-4},
        planet_varpi_rate_rad_yr={},
        great_inequality=None,
        order=1,
        a_mean_au=A_RES,
        nominal_axis_au=A_RES,
        exact_axis_au=A_RES + 0.002,
        axis_from_mean_longitude_rate_au=A_RES - 0.003,
        identity_coefficient=-1.5 * -2 * 0.79 / A_RES,
        rate_expected_rad_yr=1e-4,
        fair_step_fraction=1 / A_RES**1.5,
        crosses=(),
        synodic_periods_yr={'Jupiter': 23.7},
    )
    base.update(given)
    return make(m.Context, **base)


def _ledger():
    ledger = Ledger(thin_band=0.1, conventions={'mmr_l1_tp_full'})
    ledger.compare('L1 TP', 40, '>=', 7, '', 'geo_n_tp', setting='mmr_l1_tp_full')
    ledger.compare('L4 force R2', 0.36, '>=', 0.4, '', 'rsp_force_R2_h1', setting='mmr_force_r2_fail')
    ledger.compare('L6 shift', 0.48, '<=', 0.5, '', 'ep_centre_shift_over_amp', setting='mmr_l6')
    ledger.compare('Z zone', None, '>=', 2, '', 'sep_zone_h_lo')
    return ledger


def libration_case():
    """A two-body libration: 40 turning points, one accepted episode, an energy per cycle, FAIR drawable."""
    spec = m.ResonanceSpec('3J-2+0-1', m.ResonanceKind.TWO_BODY, ('Jupiter',), (3,), -2, -1, (0,), nominal_axis_au=A_RES)
    sigma, rate, a_f, adot = _libration_parts()
    a = a_f + 0.002 * np.sin(2 * np.pi * T / 7.9)
    series = _series(spec, sigma, a)
    cutoff = m.Cutoff(35.6, 'synodic', 35.6, 20.0, 35.6)
    prep = make(
        m.Preprocessed,
        cutoff=cutoff,
        t_yr=T,
        angle_unwrapped=sigma,
        angle_clean=sigma,
        angle_f=sigma,
        rate_f=rate,
        angle_f_half=sigma,
        angle_f_double=sigma,
        a_f=a_f,
        adot_f=adot,
        e_f=series.e,
        inc_f=series.inc,
        window=(71.2, 9928.8),
        ripple_rad=None,
    )
    max_t = 125.0 + P_LIB * np.arange(20)
    min_t = 375.0 + P_LIB * np.arange(20)
    tp_t = np.sort(np.concatenate([max_t, min_t]))
    kinds = np.where(np.isin(tp_t, max_t), 1, -1)
    tp = geo.TurningPoints(tp_t.astype(int), tp_t, np.pi + kinds * 1.0, kinds, 0.02)
    centre = geo.CentreTrack(
        T,
        np.full_like(T, np.pi),
        1500.0,
        geo.TurningPoints(np.array([], int), np.array([]), np.array([]), np.array([]), 0.1),
        [],
        0.0,
        0.0,
        1e4,
        0.0,
        0.0,
    )
    episode = make(
        geo.Episode, episode_id=1, tp_times_yr=tp_t, tp_values_rad=tp.values_rad, accepted=True, shape=geo.EpisodeShape.STATIONARY
    )
    rejected = make(geo.Episode, episode_id=2, tp_times_yr=np.array([9900.0, 9950.0]), tp_values_rad=np.array([4.0, 2.3]), accepted=False)
    geometry = make(geo.Geometry, window=prep.window, turning_points=tp, centre=centre, runs=[episode, rejected], period_cycle_yr=P_LIB)
    cycles = [pend.CycleEnergy(t0, t0 + P_LIB, False, 0.35 + 0.01 * i, NAN, P_LIB, 1.0) for i, t0 in enumerate(max_t[:-1])]
    cycles.append(pend.CycleEnergy(9625.0, 9900.0, True, 1.4, NAN, 275.0, 1.0))
    zone = pend.Zone(pend.ZoneMethod.NUMERIC, 0.7, 0.8, 1.2, 0.9, 1.1, False)
    energy = make(pend.EnergyAnalysis, potential_e=None, potential_f=None, cycles=cycles, zone=zone)
    windows = [m.ForceWindow(t0, t0 + 2000.0, 0.9, 1.26e-4, 0.0, 0.0) for t0 in np.arange(0.0, 9000.0, 1000.0)]
    response = make(m.Response, windows=windows, omega0_force_rad_yr=2 * np.pi / 480, multiplet=None)
    episode_response = make(m.EpisodeResponse, episode_id=1, before_jump=False, n_tp=0, force_r2_h1=0.91, force_r2_h2=0.88)
    physics = make(m.EpisodePhysics, episode_id=1, half_width_au=0.012, response=episode_response, omega0_rad_yr=2 * np.pi / 480)
    ledger = _ledger()
    verdict = Verdict(
        ResonanceType.MMR,
        Status.RESONANT,
        AngleStatus.LIBRATION,
        AngleSubtype.STATIONARY,
        Confidence.MEDIUM,
        Confidence.MEDIUM,
        reason=Reason.DYNAMIC_LIBRATION,
        libration_origin=LibrationOrigin.DYNAMIC,
        checks={
            'check_mmr_alias': CheckValue.PASS,
            'check_mmr_a_response': CheckValue.FAIL,
            'check_mmr_subresonances': CheckValue.NOT_APPLICABLE,
        },
        decision_path=ledger.entries,
    )
    analysis = make(
        m.MMRAnalysis,
        series_name='synthetic',
        resonance=spec,
        context=_context(),
        sampling=make(m.Sampling, angle_unwrapped=sigma, residual_rad=np.zeros(len(T) - 1)),
        prep=prep,
        slow=None,
        geometry=geometry,
        energy=energy,
        response=response,
        kinematics=None,
        subresonances=None,
        owner=None,
        strength=None,
        baseline=None,
        episode_physics=[physics],
        verdict=verdict,
        settings_hash='0123456789ab',
        confidence_cells_sha='',
    )
    return analysis, series


def circulation_case():
    """A three-body circulation: no episode, no energy, no subresonance trigger."""
    spec = m.ResonanceSpec('4J-2S-1+0+0-1', m.ResonanceKind.THREE_BODY, ('Jupiter', 'Saturn'), (4, -2), -1, -1, (0, 0))
    sigma = 2 * np.pi * T / 700.0 + 0.3 * np.sin(2 * np.pi * T / 90)
    rate = 2 * np.pi / 700.0 + 0.3 * 2 * np.pi / 90 * np.cos(2 * np.pi * T / 90)
    a = np.full_like(T, 2.6) + 1e-4 * np.sin(2 * np.pi * T / 90)
    series = _series(spec, sigma, a, planets=False)
    prep = make(
        m.Preprocessed,
        cutoff=m.Cutoff(20.0, 'floor', 15.0, 20.0, 20.0),
        t_yr=T,
        angle_unwrapped=sigma,
        angle_clean=sigma,
        angle_f=sigma,
        rate_f=rate,
        angle_f_half=sigma,
        angle_f_double=sigma,
        a_f=a,
        adot_f=np.gradient(a, T),
        e_f=series.e,
        inc_f=series.inc,
        window=(40.0, 9960.0),
        ripple_rad=None,
    )
    empty_tp = geo.TurningPoints(np.array([], int), np.array([]), np.array([]), np.array([]), 0.02)
    centre = geo.CentreTrack(T, sigma, 2100.0, empty_tp, [], 89.8, 0.0, 1e4, 14.3, 89.8)
    geometry = make(geo.Geometry, window=prep.window, turning_points=empty_tp, centre=centre, runs=[])
    verdict = Verdict(
        ResonanceType.MMR,
        Status.NON_RESONANT,
        AngleStatus.CIRCULATION,
        confidence=Confidence.HIGH,
        reason=Reason.CIRCULATION,
        checks={'check_mmr_subresonances': CheckValue.NOT_APPLICABLE, 'check_mmr_owner': CheckValue.NOT_APPLICABLE},
    )
    analysis = make(
        m.MMRAnalysis,
        series_name='circulator',
        resonance=spec,
        context=_context(a_mean_au=2.6, fair_step_fraction=0.24),
        sampling=make(m.Sampling, angle_unwrapped=sigma),
        prep=prep,
        slow=None,
        geometry=geometry,
        energy=None,
        response=None,
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
    return analysis, series


def undefined_case():
    """The unwrap failed: no filtered series, no geometry; the figure still has every panel."""
    analysis, series = circulation_case()
    analysis.prep = analysis.geometry = None
    analysis.verdict = Verdict(ResonanceType.MMR, Status.INVALID, AngleStatus.UNDEFINED, AngleSubtype.UNPHYSICAL, reason=Reason.UNPHYSICAL)
    return analysis, series


SUMMARY_NUMBERS = {
    'name': 'synthetic',
    'resonance': '3J-2+0-1',
    'smp_span_yr': 10000.0,
    'smp_step_yr': 1.0,
    'smp_n_rows': 10001,
    'flt_cutoff_yr': 35.6,
    'geo_n_tp': 40,
    'rsp_force_R2_h1': 0.36,
    'ep_centre_shift_over_amp': 0.48,
    'sep_zone_h_lo': None,
    'mrg_L1': 'full',
    'mrg_L1_value': 40.0,
    'mrg_L4': 'thin',
    'mrg_L4_value': 0.36,
}
NUMBERS = {
    'ctx_a_star_au': 3.9720004321,
    'ctx_a_mean_au': 3.97,
    'ctx_c_rad_yr_au': 0.5970,
    'smp_unwrap_resid_p99_deg': 1.2345e-3,
    'geo_period_cycle_yr': 500.0,
    'geo_brand_new_number_deg': 12.5,  # a column this file does not know: printed by its prefix
    'rsp_force_K_au_yr': 1.26e-4,
    'kin_corr_rate': float('nan'),
}


def rows_of(analysis):
    summary = {**analysis.verdict.to_flat_dict(), **SUMMARY_NUMBERS, 'name': analysis.series_name, 'resonance': analysis.resonance.key}
    episodes = [
        {
            'episode_id': 1,
            'accepted': True,
            'subtype': 'stationary',
            't_start_yr': 125.0,
            't_end_yr': 9875.0,
            'n_tp': 40,
            'period_med_yr': 500.0,
            'half_amp_min_deg': 57.3,
            'half_amp_max_deg': 57.3,
            'centre_shift_over_amp': 0.0,
        },
        {'episode_id': 2, 'accepted': False, 'reject_reason': 'too_few_tp', 't_start_yr': 9900.0, 't_end_yr': 9950.0, 'n_tp': 2},
    ]
    constants = [{'planet': 'Jupiter', 'n_deg_yr': 30.3496, 'n_source': 'measured', 'mu': 9.548e-4, 'a_mean_au': 5.2026}]
    return summary, dict(NUMBERS), episodes, constants


# --------------------------------------------------------------------------------------------- text


class TestFormatting:
    @pytest.mark.parametrize(
        'column, value, text',
        [
            ('ctx_a_star_au', 2.7526401234, '2.75264'),  # 7 significant digits for an axis: 2.752640 -> 2.75264
            ('rsp_force_K_au_yr', 1.23456e-7, '1.235e-07'),  # a rate in au/yr: 4 digits
            ('smp_n_rows', 100001.0, '100001'),  # an integral float prints as an integer
            ('geo_n_tp', 9, '9'),
            ('sep_h_med', float('nan'), '—'),
            ('flags', '', '—'),
            ('flags', 'exit_suspected;kick', 'exit_suspected; kick'),
            ('geo_bands_overlap', True, 'yes'),
        ],
    )
    def test_format_cell(self, column, value, text):
        assert panel.format_cell(column, value) == text

    @pytest.mark.parametrize(
        'column, unit, label',
        [
            ('smp_unwrap_resid_p99_deg', '°', 'unwrap_resid_p99'),
            ('rsp_force_K_au_yr', 'au/yr', 'force_K'),
            ('ctx_rate_expected_deg_yr', '°/yr', 'rate_expected'),
            ('kin_lambda_part_rate_rad_kyr', 'rad/kyr', 'lambda_part_rate'),
            ('ctx_c_rad_yr_au', 'rad/yr/au', 'c'),
            ('own_psi_share', '', 'psi_share'),
        ],
    )
    def test_units_and_labels(self, column, unit, label):
        assert panel.unit_of(column) == unit
        assert panel.label_of(column) == label


class TestTextBlocks:
    def blocks(self):
        analysis, _ = libration_case()
        summary, numbers, episodes, constants = rows_of(analysis)
        return analysis, summary, numbers, panel.text_blocks(summary, numbers, analysis.verdict.decision_path, episodes, constants)

    def test_every_number_equals_its_column(self):
        _, summary, numbers, blocks = self.blocks()
        row = {**numbers, **summary}
        printed = {line.column: line for line in cn.printed_lines(blocks, panel.TEXT_BLOCKS) if line.column and line.label}
        for column, value in row.items():
            if column.startswith(('mrg_', 'check_')) or column in ('decision_path',) or column.endswith('_value'):
                continue
            assert column in printed, column
            line = printed[column]
            if isinstance(value, float) and not math.isnan(value):
                # 4 significant digits, 7 for an axis in au (half a unit of the last digit)
                tolerance = {'abs': 5e-7} if column.endswith('_au') else {'rel': 5e-4}
                assert float(line.value) == pytest.approx(value, **tolerance), column
            elif value is None or (isinstance(value, float) and math.isnan(value)) or value == '':
                assert line.value == '—', column

    def test_unknown_column_is_printed_in_its_prefix_block(self):
        *_, blocks = self.blocks()
        lines = [line for line in blocks['geometry'].lines if line.column == 'geo_brand_new_number_deg']
        assert len(lines) == 1 and lines[0].value == '12.5' and lines[0].unit == '°' and lines[0].label == 'brand_new_number'

    def test_every_ledger_entry_is_printed_with_its_mark(self):
        analysis, *_, blocks = self.blocks()
        path = blocks['path'].lines
        assert [line.value for line in path] == [entry.describe() for entry in analysis.verdict.decision_path]
        assert [line.mark for line in path] == ['✓', '✗', '✓', '–']
        # a number decided by a ledger entry carries its mark in its own block too
        geo_tp = next(line for line in blocks['geometry'].lines if line.column == 'geo_n_tp')
        assert geo_tp.mark == '✓'

    def test_checks_and_margins(self):
        *_, blocks = self.blocks()
        checks = {line.column: (line.value, line.mark) for line in blocks['checks'].lines}
        assert len(checks) == 16
        assert checks['check_mmr_alias'] == ('pass', '✓')
        assert checks['check_mmr_a_response'] == ('fail', '✗')
        assert checks['check_mmr_subresonances'] == ('not_applicable', '–')
        margins = {line.column: line for line in blocks['margins'].lines}
        assert margins['mrg_L1_value'].label == 'L1 full' and margins['mrg_L1_value'].mark == '✓'
        assert margins['mrg_L4_value'].value == '0.36' and margins['mrg_L4_value'].mark == '~'

    def test_block_that_did_not_run_says_why(self):
        *_, blocks = self.blocks()
        assert [line.value for line in blocks['subresonances'].lines] == ['not run: no trigger']
        assert [line.value for line in blocks['owner'].lines] == ['not computed']

    def test_episode_and_constant_tables(self):
        *_, blocks = self.blocks()
        first, second = (line.value for line in blocks['episodes'].lines[:2])
        assert first == '#1 125–9875 yr 40 TP P 500 yr A_min 57.3° A_max 57.3° shift/A 0 stationary'
        assert second == '#2 9900–9950 yr 2 TP rejected: too_few_tp'
        # the planets are two aligned tables and every constants.csv column is printed (requests-w2-figure.md §2)
        constants = '\n'.join(line.value for line in blocks['constants'].lines)
        for text in ('30.35', 'meas', '9.548e-04', '5.2026'):
            assert text in constants


class TestPlacement:
    def test_blocks_keep_their_columns_when_they_fit(self):
        analysis, _ = libration_case()
        summary, numbers, episodes, constants = rows_of(analysis)
        blocks = panel.text_blocks(summary, numbers, analysis.verdict.decision_path, episodes, constants)
        placement = cn.plan_columns(blocks, mc.LAYOUT_V1, panel.TEXT_BLOCKS)
        titles = [[row.text for row in column if row.kind == 'title'] for column in placement.columns]
        # one stream through the three columns in TEXT_BLOCKS order; a block cut by a column end continues as '(cont.)'
        assert titles[0][0] == 'VERDICT'
        flowing = [title for column in titles for title in column if not title.endswith('(cont.)')]
        expected = [panel.BLOCKS[name].title for name in panel.TEXT_BLOCKS]
        assert flowing == [title for title in expected if title in flowing]
        assert placement.overflow == 0

    def test_nothing_is_cut(self):
        """600 numbers do not fit at 7.5 pt: the text flows and the font shrinks, every line stays."""
        numbers = {f'geo_extra_{i:03d}_deg': i + 0.5 for i in range(600)}
        blocks = panel.text_blocks({}, numbers, [])
        placement = cn.plan_columns(blocks, mc.LAYOUT_V1, panel.TEXT_BLOCKS)
        assert placement.font_pt < mc.LAYOUT_V1.text_font_pt
        cells = [cell.label for column in placement.columns for row in column for cell in row.cells]
        assert all(f'extra_{i:03d}' in cells for i in range(600))
        stats = cn.place_blocks(blocks, mc.LAYOUT_V1, panel.TEXT_BLOCKS)
        assert stats['overflow'] == 0 and max(stats[f'column_{i}'] for i in (1, 2, 3)) <= stats['capacity']


# --------------------------------------------------------------------------------------------- figure


ALL_NUMBERS = {'combined': {'numbers': 'all'}}  # the right column as the full dump (the 'explained' form: test_verdict_text)


def _draw(case, options=None):
    analysis, series = case()
    return analysis, mc.draw_combined(analysis, series, rows=rows_of(analysis), options=options)


def _texts(fig):
    return [text.get_text() for text in fig.findobj(matplotlib.text.Text)]


def _rects(fig):
    return [tuple(np.round(ax.get_position().bounds, 6)) for ax in fig.axes]


class TestFigure:
    def teardown_method(self):
        plt.close('all')

    def test_same_size_and_layout_for_every_object(self):
        """Libration with FAIR, circulation without, undefined without anything: the same rectangles."""
        figures = [_draw(case)[1] for case in (libration_case, circulation_case, undefined_case)]
        width, height = mc.LAYOUT_V1.size
        x, y, w, h = mc.LAYOUT_V1.panels
        n = mc.LAYOUT_V1.n_panels
        gap = mc.LAYOUT_V1.panel_gap
        step = (h - (n - 1) * gap) / n  # ten panels with a gap between neighbours
        expected = [(x / width, (y + h - (i + 1) * step - i * gap) / height, w / width, step / height) for i in range(n)]
        for fig in figures:
            assert tuple(fig.get_size_inches()) == (width, height)
            panels = [tuple(ax.get_position().bounds) for ax in fig.axes[:n]]
            assert np.allclose(panels, expected)
            assert all(b[1] + b[3] < a[1] for a, b in zip(panels, panels[1:]))  # neighbours do not touch
        # e and i on two panels (no twin axis): equal rectangles; only FAIR's slot is empty without FAIR (the legend
        # takes it, author's P10)
        x3, y3, w3, h3 = mc.LAYOUT_V1.bottom[2]
        fair = tuple(np.round((x3 / width, y3 / height, w3 / width, h3 / height), 6))
        assert fair in _rects(figures[0])
        assert [rect for rect in _rects(figures[0]) if rect != fair] == _rects(figures[1]) == _rects(figures[2])

    def test_every_ledger_column_and_entry_is_in_the_text(self):
        analysis, fig = _draw(libration_case, ALL_NUMBERS)
        texts = _texts(fig)
        for entry in analysis.verdict.decision_path:
            assert entry.describe() in texts
        # the status is printed once, in the plaque of the right column (author, 2026-10-16b)
        _, fig = _draw(libration_case)
        plaques = [text for text in _texts(fig) if 'RESONANT · libration' in text]
        assert len(plaques) == 1 and 'MEDIUM' in plaques[0]

    def test_reasons_for_what_did_not_run(self):
        _, fig = _draw(circulation_case, ALL_NUMBERS)
        texts = ' '.join(_texts(fig))
        assert 'FAIR no (three-body angle)' in texts  # the record line; the slot of FAIR holds the legend
        assert '(k) phase portrait, whole record' in _texts(fig) and '(m) FAIR' not in _texts(fig)
        # why a check did not run is in the right column, not on the panels (author 16b; W8): 'SUBRESONANCES – not run'
        assert 'SUBRESONANCES' in _texts(fig) and 'not run: no trigger' in _texts(fig)
        assert 'subresonances not run' not in texts
        assert 'h not computed: no energy analysis' in texts

    def test_fair_drawn_for_prograde_two_body(self):
        analysis, series = libration_case()
        assert mc.fair_unavailable_reason(analysis, series) is None
        analysis.context.fair_step_fraction = 0.3  # step 0.3 P_orb > 1/4.2
        assert mc.fair_unavailable_reason(analysis, series).startswith('step 0.3')

    def test_marks_are_coloured(self):
        _, fig = _draw(libration_case, ALL_NUMBERS)
        colours = {
            text.get_text(): matplotlib.colors.to_hex(text.get_color())
            for text in fig.findobj(matplotlib.text.Text)
            if text.get_text() in ('✓', '✗', '~')
        }
        assert colours == {'✓': '#1b7837', '✗': '#b2182b', '~': '#e08214'}

    def test_stamp_is_printed_and_plotter_saves(self, tmp_path):
        analysis, series = libration_case()
        summary, numbers, episodes, constants = rows_of(analysis)
        analysis.summary_row = lambda: summary
        analysis.numbers_row = lambda: numbers
        analysis.episode_rows = lambda: episodes
        analysis.context.constants_rows = lambda: constants
        plotter = mc.VerdictFigurePlotter(analysis, series)
        fig = plotter.plot(stamp='recomputed != summary')
        assert 'recomputed != summary' in _texts(fig)
        plotter.save(tmp_path / 'v1.png')
        plotter.close()
        assert (tmp_path / 'v1.png').stat().st_size > 0

    def test_plaque_and_final_regime_tag(self):
        """status · angle_status · confidence in the plaque (author, 2026-10-17); a record that ends in another regime
        gets the tag 'ends: …' right of the plaque, in the status colour, on the plaque's middle line."""
        from resonances.plotting.style import STATUS_COLOURS

        analysis, series = libration_case()
        fig = mc.draw_combined(analysis, series, rows=rows_of(analysis))
        assert ' RESONANT · libration (stationary) · MEDIUM ' in _texts(fig)
        assert not any('ends:' in text for text in _texts(fig))
        plt.close(fig)

        analysis.verdict.final_regime = 'circulation'
        fig = mc.draw_combined(analysis, series, rows=rows_of(analysis))
        fig.canvas.draw()  # places the boxes
        renderer = fig.canvas.get_renderer()
        artists = {text.get_text(): text for text in fig.texts}
        plaque, tag = artists[' RESONANT · libration (stationary) · MEDIUM '], artists[' ends: circulation ']
        box_p, box_t = plaque.get_bbox_patch().get_window_extent(renderer), tag.get_bbox_patch().get_window_extent(renderer)
        assert box_t.x0 > box_p.x1  # beside the plaque, not over it
        assert abs((box_t.y0 + box_t.y1) / 2 - (box_p.y0 + box_p.y1) / 2) < 0.15 * box_p.height
        colour = STATUS_COLOURS['resonant']
        assert matplotlib.colors.to_hex(tag.get_color()) == colour
        assert matplotlib.colors.to_hex(tag.get_bbox_patch().get_edgecolor()) == colour

    def test_rate_zero_and_portrait_centres(self):
        """(c) and both portraits draw dsigma/dt = 0 in the style RATE_ZERO; the portraits mark the accepted episode's
        centre at (sigma_c, 0): 100 deg in the 360-degree window from 30 deg is 100, from 150 deg it is 460."""
        from resonances.plotting.style import RATE_ZERO

        analysis, series = libration_case()
        analysis.geometry.runs[0].centre_mod_2pi_rad = np.radians(100.0)
        fig = mc.draw_combined(analysis, series, rows=rows_of(analysis))

        def zero_lines(ax):
            return [
                line
                for line in ax.get_lines()
                if np.allclose(line.get_ydata(), 0.0)
                and line.get_linestyle() == RATE_ZERO['ls']
                and matplotlib.colors.to_hex(line.get_color()) == RATE_ZERO['color']
                and line.get_linewidth() == RATE_ZERO['lw']
            ]

        assert zero_lines(fig.axes[2])  # (c)
        portraits = [ax for ax in fig.axes if any(t.get_text().startswith(('(k)', '(l)')) for t in ax.texts)]
        assert len(portraits) == 2
        for ax in portraits:
            assert zero_lines(ax)
            marks = [line for line in ax.get_lines() if line.get_marker() == '+']
            assert len(marks) == 1
            low = ax.get_xlim()[0]
            expected = 100.0 if low <= 100.0 < low + 360 else 100.0 + 360 * np.floor((low + 360 - 100.0) / 360)
            assert np.allclose(marks[0].get_xdata(), [expected]) and np.allclose(marks[0].get_ydata(), [0.0])
        plt.close(fig)

        ax = plt.figure().add_subplot()
        mc.centre_marks(ax, [make(geo.Episode, episode_id=1, centre_mod_2pi_rad=np.radians(100.0))], np.radians(150.0))
        assert np.allclose(ax.get_lines()[0].get_xdata(), [460.0])


class TestCentrePanelModes:
    """(j) shows the subresonance, the owner or the kinematic terms when the analysis found them: their lines are
    labelled at the right by key, in their colour; no caption (author, 2026-10-16)."""

    def teardown_method(self):
        plt.close('all')

    def labels(self, analysis, series):
        """{text: colour} of the labels of panel (j); asserts there is no caption."""
        fig = plt.figure(figsize=mc.LAYOUT_V1.size, dpi=mc.LAYOUT_V1.dpi)
        panel_j = mc.draw_time_panels(fig, analysis, series, mc.LAYOUT_V1)[-1]
        assert not [text for text in panel_j.texts if text.get_text().startswith('mode:')]
        return {text.get_text(): matplotlib.colors.to_hex(text.get_color()) for text in panel_j.texts if text.get_position()[0] > 1}

    def test_subresonance(self):
        analysis, series = libration_case()
        sub_spec = m.ResonanceSpec('3J-2+2-3', m.ResonanceKind.TWO_BODY, ('Jupiter',), (3,), -2, -3, (2,))
        accepted = make(m.SubresonanceCandidate, spec=sub_spec, j=2, accepted=True)
        analysis.subresonances = make(m.SubresonanceResult, accepted=accepted, check=CheckValue.FAIL)
        analysis.derived = [libration_case()[0]]
        assert mc.centre_mode(analysis) == 'subresonance'
        assert self.labels(analysis, series)['3J-2+2-3'] == mc.SUBRESONANCE

    def test_owner(self):
        analysis, series = libration_case()
        owner_spec = m.ResonanceSpec('11M-20+0+9', m.ResonanceKind.TWO_BODY, ('Mars',), (11,), -20, 9, (0,))
        accepted = make(m.OwnerCandidate, spec=owner_spec, r2_median=0.95, strength_ratio=1.2, accepted=True)
        analysis.owner = make(m.OwnerResult, accepted=accepted, check=CheckValue.FAIL)
        assert mc.centre_mode(analysis) == 'owner'
        # no Mars longitudes in the record: the owner's centre cannot be drawn, only sigma0's (the owner is in the column)
        labels = self.labels(analysis, series)
        assert labels['3J-2+0-1'] == mc.CENTRE and '11M-20+0+9' not in labels

    def test_kinematic(self):
        analysis, series = libration_case()
        analysis.verdict.libration_origin = LibrationOrigin.KINEMATIC
        analysis.verdict.origin_detail = 'varpi_term'
        analysis.slow = make(
            m.SlowComponents,
            window_yr=1500.0,
            t_yr=T,
            angle_slow=np.full_like(T, np.pi),
            varpi_slow=series.varpi,
            window=(1500.0, 8500.0),
        )
        analysis.kinematics = make(m.Kinematics, corr_rate=0.991)
        assert mc.centre_mode(analysis) == 'kinematic'
        assert self.labels(analysis, series) == {'σ': mc.CENTRE, '-1ϖ': mc.SUBRESONANCE}
