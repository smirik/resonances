"""The slotted explanation of the secular/ZLK figure (W28 E9; cards g09-5 C, g10-8, g10-4b, g09-3, g09-6, g09-7).

The right column of a secular or Lidov-Kozai figure reads as bullets in the slots of the MMR one (verdict, what the
angle does, the decisive argument, objections, cause) plus the time scales of the record: 1/T, |σ̇|/(1/T), the search
window, the periods seen on the angle with the FREQUENCIES row they are, and one sentence that planetary periods are
precession times. A number a bullet prints leaves the numbers table for a pointer (W20 CS11).

Expectations by hand: a period of P yr is 1 296 000 / P ″/yr; 1/T of a 2-Myr record is 0.648″/yr, of a 500-kyr record
2.592″/yr. The synthetic secular angle g − g6 + s − s6 has g = 57.8, s = −56.1″/yr, so σ̇ = 57.8 − 28.2455 − 56.1 +
26.3450 = −0.2005″/yr and |σ̇|/(1/T) = 0.309; its filter cutoff is P_c = 100 kyr (the secular shortest libration
200 kyr over FILTER_MIN_LIBRATION_CUTOFFS = 2), so the window runs from 2 P_c = 200 kyr to T − 2 P_c = 1.8 Myr. The
synthetic ZLK body (591986's analogue) has a free circle at 222″/yr about a forced vector at g5: its ω cycle is the beat
1 296 000 / (222 − 4.2575) = 5.95 kyr = the row g − g5, and its free ω turns (222 − 4.31) × 500 000 / 1 296 000 = 84
times. The 1981 Midas numbers of the node test are D8's measurement (|b_Ω − s6| 1.85″/yr, resolution 2.592″/yr,
separation 700 kyr). Benchmark records (skipped without the data) check the card examples.
"""

import re
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from resonances.data.const import PLANETARY_FREQUENCIES  # noqa: E402
from resonances.plotting import verdict_text as vt  # noqa: E402
from resonances.plotting.combined_numbers import NumberLine  # noqa: E402
from resonances.plotting.secular_plots import Frequency, draw_long_period_combined, point_to_text  # noqa: E402
from resonances.resonance.analysis.secular import AngleTerm, LongPeriodSeries, analyse_secular  # noqa: E402
from resonances.resonance.analysis.settings import settings_from_config  # noqa: E402
from resonances.resonance.analysis.verdict import AngleSubtype, AngleStatus, Reason, ResonanceType, Status  # noqa: E402
from resonances.resonance.analysis.zlk import analyse_zlk  # noqa: E402
from resonances.secular import free_elements as fe  # noqa: E402

SETTINGS = settings_from_config()
RAD = 2 * np.pi / 1296000.0  # rad/yr per arcsec/yr
G5, G6, S6 = PLANETARY_FREQUENCIES['g5'], PLANETARY_FREQUENCIES['g6'], PLANETARY_FREQUENCIES['s6']
NUMBER = r'([0-9]+(?:\.[0-9]+)?)'


def _bullet(text: vt.VerdictText, slot: str, start: str = '') -> str:
    """The text of the first bullet of `slot` (that starts with `start`), '' when there is none."""
    return next((b.text for b in text.bullets if b.slot == slot and b.text.startswith(start)), '')


def _all(text: vt.VerdictText) -> str:
    return ' | '.join([b.text for b in text.bullets] + list(text.sentences) + list(text.info))


@pytest.fixture(scope='module')
def secular():
    """2306-like g − g6 + s − s6 over 2 Myr (tests/resonances/plotting/test_long_period_frequencies.py)."""
    t = np.linspace(0.0, 2e6, 4001)
    g, s = 57.8, -56.1
    varpi, Omega = g * RAD * t + 0.3, s * RAD * t + 1.1
    sigma = varpi - G6 * RAD * t + Omega - S6 * RAD * t
    n = len(t)
    terms = (AngleTerm('g', None, 1), AngleTerm('g', 6, -1), AngleTerm('s', None, 1), AngleTerm('s', 6, -1))
    series = LongPeriodSeries(
        ResonanceType.SECULAR,
        'g-g6+s-s6',
        'synthetic',
        t,
        sigma,
        2.73 + 0.002 * np.sin(2 * np.pi * t / 3e4),
        np.full(n, 0.05),
        np.full(n, np.radians(5.0)),
        varpi=varpi,
        Omega=Omega,
        terms=terms,
        planet_frequencies=PLANETARY_FREQUENCIES,
    )
    analysis = analyse_secular(series, SETTINGS)
    return analysis, analysis.summary_row()


@pytest.fixture(scope='module')
def zlk():
    """591986's analogue over 500 kyr: forced 0.045 at g5 and 0.017 at g6, the free circle 0.011 at 222″/yr, the node
    at g5, i 39° (a kinematic libration of ω)."""
    t = np.arange(0.0, 500_001.0, 100.0)
    z_e = 0.045 * np.exp(1j * (G5 * RAD * t + np.radians(83.0))) + 0.017 * np.exp(1j * (G6 * RAD * t + np.radians(3.2)))
    z_e = z_e + 0.011 * np.exp(1j * 222.0 * RAD * t)
    Omega = G5 * RAD * t + np.radians(146.0) + np.radians(8.0) * np.sin(2 * np.pi * t / 1.5e6)
    e, varpi = np.abs(z_e), np.angle(z_e)
    inc = np.full_like(t, np.radians(39.0))
    omega = np.unwrap(varpi - Omega)
    series = LongPeriodSeries(ResonanceType.LIDOV_KOZAI, 'LK', 'synthetic', t, omega, 5.2 + 0.05 * np.sin(2 * np.pi * t / 1e4), e, inc)
    elements = fe.free_elements(t, e, inc, Omega, omega)
    analysis = analyse_zlk(series, SETTINGS, free_elements=elements)
    row = {**analysis.summary_row(), **fe.summary_fields(elements)}
    return analysis, row, elements


def _explain(analysis, row, elements=None):
    from resonances.plotting.secular_plots import long_period_frequencies

    return vt.explain_long_period(analysis, SETTINGS, analysis.episode_rows(), row, long_period_frequencies(analysis, row, elements))


# --------------------------------------------------------------------------------------------- time scales


def test_the_time_scales_of_a_secular_angle(secular):
    analysis, row = secular
    assert analysis.angle.cutoff_yr == pytest.approx(1e5)  # P_c, as the module docstring derives it
    scale = _bullet(_explain(analysis, row), vt.SLOT_SCALE)
    assert '1/T = 0.648″/yr (T = 2 Myr)' in scale
    ratio = float(re.search(rf'\|σ̇\|/\(1/T\) = {NUMBER}', scale).group(1))
    assert ratio == pytest.approx(0.2005 / 0.648, abs=0.02) and 'turns less than once' in scale
    assert 'search window 200 kyr–1.8 Myr (from 2 P_c' in scale
    assert scale.endswith(vt.PRECESSION_SENTENCE)


def test_a_zlk_row_gets_its_physics_and_its_period_named(zlk):
    analysis, row, elements = zlk
    assert analysis.verdict.reason == 'kinematic_libration'
    text = _explain(analysis, row, elements)
    assert _bullet(text, vt.SLOT_VERDICT) == 'NOT A LIDOV-KOZAI LIBRATION: ω librates, but the free ω winds.'
    physics = _bullet(text, vt.SLOT_CAUSE, 'Physics:')
    assert 'i 39–39°' in physics and '39.2' not in physics  # no quadrupole reference outside its domain (W30)
    turns = float(re.search(rf'turns {NUMBER} times', physics).group(1))
    assert turns == pytest.approx((222.0 - 4.31) * 5e5 / 1296000, rel=0.01)
    assert 'Induced by the forced eccentricity of mode g5' in _bullet(text, vt.SLOT_CAUSE, 'Induced')
    scale = _bullet(text, vt.SLOT_SCALE)
    assert '1/T = 2.59″/yr (T = 500 kyr)' in scale and '|σ̇|' not in scale  # ZLK: the free-ω turns say it
    # g − s = 222 − 4.31 and g − g5 = 222 − 4.2575 differ by far less than 1/T: both rows are named
    named = re.search(r'the period of the deciding episode \(above\) ≈ ([^.;]+)\.', scale).group(1).split(' or ')
    assert {'g − s', 'g − g5'} <= set(named)
    assert 'beat period mismatch' not in _all(text)


def test_identify_names_a_row_within_one_over_t():
    """g5 turns in 1296000/4.2575 = 304.4 kyr: a period of 300 kyr (290–310) over 1 Myr (1/T 1.296″/yr) is g5; 2306's
    221 kyr (218–257 kyr: 5.04–5.94″/yr, 1/T 0.13″/yr over 10 Myr) is none of g5, g6, s6, and g5 is the nearest."""
    rows = [Frequency('g5 Jup', G5, 1296000 / G5), Frequency('g6 Sat', G6, 1296000 / G6), Frequency('s6 Sat', S6, 1296000 / abs(S6))]
    assert vt._identify(3.0e5, 2.9e5, 3.1e5, 1.296, rows) == ('g5 Jup', 'g5 Jup')
    # 46.5 kyr (27.87″/yr) over 100 kyr (1/T 12.96″/yr): g6 (28.25) and s6 (26.34) are both within 1/T, nearest first
    assert vt._identify(4.65e4, float('nan'), float('nan'), 12.96, rows) == ('g6 Sat or s6 Sat', 'g6 Sat')
    assert vt._identify(2.21e5, 2.185e5, 2.57e5, 0.1296, rows) == ('', 'g5 Jup')
    assert vt._identify(1296000 / 26.0, float('nan'), float('nan'), 0.1296, rows) == ('', 's6 Sat')  # 26.000 vs 26.345


# --------------------------------------------------------------------------------------------- the decisive argument


def test_the_node_near_s6_is_the_reason_not_the_cycles(zlk):
    analysis, row, elements = zlk
    gate = replace(analysis.gate, outcome='planetary_residual')
    verdict = replace(analysis.verdict, status=Status.UNCERTAIN, reason=Reason.GATE_UNAVAILABLE)
    node = {
        **row,
        'zlk_gate_node_near': 's6',
        'zlk_gate_node_near_d_arcsec_yr': 1.850892627016151,
        'zlk_gate_node_near_resolution_arcsec_yr': 2.592,
        'zlk_gate_node_separation_yr': 700202.69,
    }
    text = _explain(replace(analysis, gate=gate, verdict=verdict), node, elements)
    decisive = _bullet(text, vt.SLOT_DECISIVE)
    assert 'not yet separated from s6' in decisive and '|b_Ω − s6| 1.85″/yr < 2.59″/yr (1/T' in decisive
    assert 'they separate after 700 kyr' in decisive and 'too few cycles' not in _all(text)
    assert 'so 1/T is the resolution above' in _bullet(text, vt.SLOT_SCALE)  # the 2.59 printed once


def test_a_coorbital_without_the_split_says_the_model_does_not_apply(zlk):
    analysis, row, elements = zlk
    gate = replace(analysis.gate, outcome='unavailable', detail='coorbital_unfitted_forcing')
    verdict = replace(
        analysis.verdict,
        status=Status.UNCERTAIN,
        angle_status=AngleStatus.EPISODIC,
        reason=Reason.GATE_UNAVAILABLE,
        rules_fired=['zlk_coorbital_episodic'],
    )
    text = _explain(
        replace(analysis, gate=gate, verdict=verdict), {**row, 'zlk_gate_matches': 'coorbital_unfitted_forcing:Earth'}, elements
    )
    assert 'the free-element model does not apply to a co-orbital of Earth' in _bullet(text, vt.SLOT_DECISIVE)
    assert _bullet(text, vt.SLOT_VERDICT) == 'UNCERTAIN: ω stays bounded, but the free-ω gate gives no verdict.'


def test_a_swing_within_the_phase_uncertainty(secular):
    """5026's numbers (D5): a swing of 21.9° against 33.9° under linear phases."""
    analysis, row = secular
    verdict = replace(analysis.verdict, angle_subtype=AngleSubtype.PHASE_UNRESOLVED)
    numbers = {**row, 'sec_phase_swing_deg': 21.88473, 'sec_phase_uncertainty_deg': 33.86999}
    text = _explain(replace(analysis, verdict=verdict), numbers)
    decisive = _bullet(text, vt.SLOT_DECISIVE)
    assert decisive.startswith('The swing 21.9° ≤ 33.9°, the phase uncertainty of g6, s6 (mostly ')
    assert 'SECULAR_PLANET_PHASES=measured' in decisive
    assert _bullet(text, vt.SLOT_VERDICT) == 'UNCERTAIN: the swing is within the error of the linear mode phases.'
    assert {'sec_phase_swing_deg', 'sec_phase_uncertainty_deg'} <= set(vt.bullet_columns(text))


def test_a_centre_that_turned_back_is_not_a_drift(secular):
    analysis, row = secular
    verdict = replace(analysis.verdict, rules_fired=['centre_turned_insufficient'])
    decisive = _bullet(_explain(replace(analysis, verdict=verdict), row), vt.SLOT_DECISIVE)
    assert decisive.startswith('The center turned back within less than one revolution') and 'not a drift' in decisive


def test_objections_leave_the_grey_line(secular):
    """A failed response and a single center estimate are objections (card g10-8), not gray 'fail' lines."""
    analysis, row = secular
    verdict = replace(
        analysis.verdict, angle_status=AngleStatus.LIBRATION, checks={**analysis.verdict.checks, 'check_secular_e_response': 'fail'}
    )
    text = _explain(replace(analysis, verdict=verdict), {**row, 'ep_centre_estimates': 1})
    objection = _bullet(text, vt.SLOT_OBJECTION)
    assert objection.startswith(vt.OBJECTION_HEAD) and 'do not follow the angle' in objection  # e and sin(i/2)
    assert 'only one independent center estimate' in objection
    assert not any('fail' in line for line in text.info)


def test_the_bullets_follow_the_slots(secular, zlk):
    for analysis, row, *rest in (secular, zlk):
        slots = [b.slot for b in _explain(analysis, row, *rest).bullets]
        # W34: the minimal policy opens the explanation with its four bullets, the verdict line follows
        assert slots[:5] == [vt.SLOT_POLICY] * 4 + [vt.SLOT_VERDICT] and slots[-1] == vt.SLOT_ORBIT and vt.SLOT_SCALE in slots
        assert [vt.SLOTS.index(slot) for slot in slots] == sorted(vt.SLOTS.index(slot) for slot in slots)


# --------------------------------------------------------------------------------------------- the figure


def test_a_number_of_the_text_points_up_in_the_table():
    lines = [
        NumberLine('numbers', 'i range', '39–41', '°', 'ctx_inc_min_deg', also=('ctx_inc_max_deg',)),
        NumberLine('numbers', 'span', '500000', 'yr', 'smp_span_yr'),
        NumberLine('numbers', 'step', '100', 'yr', 'smp_step_yr'),
        NumberLine('numbers', '', 'g5 Jup +4.2575 304 kyr', group='F', keep=True),
    ]
    out = point_to_text(lines, ('ctx_inc_max_deg', 'smp_span_yr'))
    assert [(line.label, line.value, line.unit) for line in out[:3]] == [
        ('i range', vt.TEXT_POINTER, ''),
        ('span', vt.TEXT_POINTER, ''),
        ('step', '100', 'yr'),
    ]
    assert out[3] == lines[3]


def test_the_figure_draws_the_column(zlk):
    """The figure draws the ten-block column (`verdict_column`, 2026-11-04) that the package export builds
    (`verdict_column_long.figure_column`): the same column, its status word and Why drawn, no '↑' pointer (D1)."""
    from resonances.plotting import verdict_column as vc
    from resonances.plotting.verdict_column_long import figure_column

    analysis, row, elements = zlk
    extra = {key: value for key, value in row.items() if key.startswith('free_')}
    fig = draw_long_period_combined(analysis, row=extra, free_elements=elements, settings=SETTINGS)
    try:
        column = fig.text_stats['column']
        assert column == figure_column(analysis, extra, elements, SETTINGS) and vc.problems(column) == []
        assert fig.text_stats['overflow'] == 0 and fig.text_stats['text'].bullets  # the VerdictText it was built from
        drawn = ' '.join(t.get_text() for t in fig.texts if t.get_gid() == 'verdict_column')
        assert column.header.word in drawn and vc.plain(column.why.sentences[0]).split()[0] in drawn
        assert vt.TEXT_POINTER not in drawn and '↑' not in drawn
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------------------------- benchmark records


def _benchmark(example: str):
    """(analysis, text) of a secular/ZLK benchmark record, analyzed as `reclassify_secular_zlk` does."""
    tools = Path(__file__).resolve().parents[3] / 'benchmark' / 'tools'
    sys.path.insert(0, str(tools))
    try:
        import bench_io as bio
        import reclassify_secular_zlk as reclassify
    except ImportError:
        pytest.skip('benchmark tools not available')
    if not bio.example_dir(example).exists():
        pytest.skip('benchmark data not available')
    sim, body, resonance = reclassify.build_simulation(bio.load_example(example, with_planets=False))
    sim.identify_librations()
    analysis = sim.long_period_analysis(body, resonance)
    extra = reclassify._extra_columns(reclassify.summary_row(example, body, resonance))
    row = {**analysis.summary_row(), **extra}
    elements = body.free_elements if resonance.type == 'lidov_kozai' else None
    return analysis, row, _explain(analysis, row, elements)


@pytest.mark.benchmark
def test_555527_reads_its_inclination_and_free_omega_turns():
    analysis, row, text = _benchmark('zlk__555527__LK')
    physics = _bullet(text, vt.SLOT_CAUSE, 'Physics:')
    assert re.search(r'i [0-9.]+–[0-9.]+°', physics) and '39.2' not in physics  # no quadrupole reference (W30)
    turns = float(re.search(rf'turns {NUMBER} times', physics).group(1))
    assert turns == pytest.approx(abs(row['free_b_omega']) * 1e5 / 1296000, rel=0.01) and turns > 5  # D1: about 8
    assert 'beat period mismatch' not in _all(text)


@pytest.mark.benchmark
def test_5026_is_below_the_resolution_with_one_centre():
    analysis, row, text = _benchmark('secular__5026__g+g2-g4-g6')
    ratio = float(re.search(rf'\|σ̇\|/\(1/T\) = {NUMBER}', _bullet(text, vt.SLOT_SCALE)).group(1))
    assert ratio == pytest.approx(0.0063 / 0.1296, abs=0.01)  # card g09-3: divisor 0.0063″/yr, 1/T 0.13″/yr
    assert 'only one independent center' in _bullet(text, vt.SLOT_OBJECTION)
    assert 'mostly g4' in _bullet(text, vt.SLOT_DECISIVE)


@pytest.mark.benchmark
def test_2306_time_scales():
    analysis, row, text = _benchmark('secular__2306__g-g6+s-s6')
    scale = _bullet(text, vt.SLOT_SCALE)
    assert '1/T' in scale and '|σ̇|/(1/T) = 1.24' in scale and 'search window' in scale
    assert re.search(r'seen on the angle: 2[0-9]{2} kyr \(its turning points\)(: unidentified| ≈)', scale)
    assert 'not a drift' in _bullet(text, vt.SLOT_DECISIVE)


@pytest.mark.benchmark
def test_pp1_counts_its_librating_episodes_only():
    analysis, row, text = _benchmark('zlk__2020_PP1__LK')
    sigma = _bullet(text, vt.SLOT_SIGMA)
    n = int(row['sec_n_librating_episodes'])
    accepted = sum(1 for episode in analysis.episode_rows() if episode['accepted'])
    assert accepted > n and f'librates in {n} separate episodes' in sigma and f'{accepted - n} more accepted run' in sigma
    assert 'the deciding one 8.08 kyr–18.1 kyr' in sigma  # card g09-6: the deciding episode, not 283–329 kyr
    assert 'model does not apply to a co-orbital of Earth' in _bullet(text, vt.SLOT_DECISIVE)


@pytest.mark.benchmark
def test_1981_names_s6_the_distance_and_the_resolution():
    analysis, row, text = _benchmark('zlk__1981__LK')
    decisive = _bullet(text, vt.SLOT_DECISIVE)
    assert 'not yet separated from s6' in decisive and '1.85″/yr < 2.59″/yr' in decisive and 'resolution' in decisive
    assert 'too few cycles' not in _all(text)


@pytest.mark.benchmark
def test_88899_500kyr_objection_reads_the_deciding_episode():
    """W34 (the W28 defect S4): the level LOW comes from the deciding episode's e-i integral (flag
    ei_integral_not_held), not from the record's coupling; the objection prints the episode's numbers."""
    analysis, row, text = _benchmark('zlk__88899__LK__500kyr')
    objection = _bullet(text, vt.SLOT_OBJECTION)
    assert 'not held in the deciding episode #1' in objection and 'η_H 1.24 > 0.5' in objection
    assert 'so the level is low' in objection and 'the e-i line below)' not in objection
    eta = float(row['zlk_ei_episode_eta_h'])
    assert f'η_H {vt.fmt(eta)} > {vt.fmt(SETTINGS.zlk_ei_h_residual_max)}' in objection


@pytest.mark.benchmark
def test_1981_says_why_the_level_is_certain():
    analysis, row, text = _benchmark('zlk__1981__LK')
    line = _bullet(text, vt.SLOT_DECISIVE, 'Why no cap')
    assert f'ρ_max {vt.fmt(float(row["zlk_gate_rho_max"]))} ≥ 3' in line and 'kinematic_excluded' in line and 'CERTAIN' in line


@pytest.mark.benchmark
def test_2010_hy22_says_why_it_is_invalid():
    analysis, row, text = _benchmark('zlk__2010_HY22__LK')
    assert _bullet(text, vt.SLOT_VERDICT) == 'INVALID: the orbit is unphysical: e reaches 2.11 > 1 (hyperbolic); no angle is read.'
    assert 'turns' not in _all(text).split('SENTENCES')[0] and not _bullet(text, vt.SLOT_SCALE)
