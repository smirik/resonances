"""The ZLK columns that change no status, and the guards of the free-omega gate (W25 ZLK iteration).

Expectations are computed independently of the package: analytic synthetic vectors (Lagrange's laws of the two
conserved momenta, a forced vector of two modes plus a free circle), hand-computed Lidov integrals, and the numbers
the two independent studies measured on the committed fixtures (`tests/fixtures/free-omega/`, 591986: the 6-kyr cycle
is a forced beat, i in phase with e at 0.0041 deg against 0.0037 predicted; its centre swings +-22 deg at the g6-g5 beat
54.0 kyr; 3040/162474/4257: ZLK coupling, phase 177-181 deg, r about -0.995).
"""

import gzip
import pathlib

import numpy as np
import pandas as pd
import pytest

from resonances.data.const import INVARIABLE_PLANE, PLANETARY_FREQUENCIES
from resonances.data.invariable_plane import inclination_to_invariable_plane, rotation_to_invariable
from resonances.lidov_kozai.lidov_kozai_resonance import LidovKozaiParameters
from resonances.resonance import omega_free_gate
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.secular import LongPeriodSeries, decide
from resonances.resonance.analysis.secular.models import GateResult
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, CheckValue, Reason, ResonanceType, Status
from resonances.resonance.analysis.zlk import analyse_zlk, diagnostics
from resonances.secular import free_elements as fe
from resonances.simulation.config import SimulationConfig

SETTINGS = settings_from_config()
AS = 1296000.0
RAD = 2 * np.pi / AS  # rad/yr per arcsec/yr
G5, G6 = PLANETARY_FREQUENCIES['g5'], PLANETARY_FREQUENCIES['g6']
I_INV, NODE_INV = np.radians(INVARIABLE_PLANE['inc']), np.radians(INVARIABLE_PLANE['Omega'])
FIXTURES = pathlib.Path(__file__).parents[2] / 'fixtures' / 'free-omega'


# ------------------------------------------------------------------------------------------- synthetic helpers


def zlk_series(t, z_e, Omega, inc, a=5.2):
    """A LongPeriodSeries of omega = arg(z_e) - Omega with the elements of the vectors."""
    e, varpi = np.abs(z_e), np.angle(z_e)
    omega = np.unwrap(varpi - Omega)
    series = LongPeriodSeries(
        ResonanceType.LIDOV_KOZAI, 'LK', 'synthetic', t, omega, np.full_like(t, a), e, inc, varpi=np.unwrap(varpi), Omega=Omega
    )
    return series, fe.free_elements(t, e, inc, Omega, omega)


def double_scale(node_offset_widths=0.0, a=5.2):
    """591986's analogue (Claude study S1): forced 0.045 at g5 + 0.017 at g6, a free circle 0.011 at 222"/yr, the node
    at g5 (+ an offset in Rayleigh widths of 500 kyr), i fixed at 39 deg."""
    t = np.arange(0.0, 500_001.0, 100.0)
    z_e = 0.045 * np.exp(1j * (G5 * RAD * t + np.radians(83.0))) + 0.017 * np.exp(1j * (G6 * RAD * t + np.radians(3.2)))
    z_e = z_e + 0.011 * np.exp(1j * 222.0 * RAD * t)
    rate = G5 + node_offset_widths * AS / 500_000.0
    Omega = rate * RAD * t + np.radians(146.0) + np.radians(8.0) * np.sin(2 * np.pi * t / 1.5e6)
    return zlk_series(t, z_e, Omega, np.full_like(t, np.radians(39.0)), a=a)


def genuine_librator(a=2.5):
    """A ZLK librator (study S2): free omega = 90 + 25 sin(2 pi t / 40 kyr), e_free 0.30 + 0.04 cos, its own node at
    -20"/yr, small forced terms at g5 and g6."""
    t = np.arange(0.0, 500_001.0, 100.0)
    omega_free = np.radians(90.0) + np.radians(25.0) * np.sin(2 * np.pi * t / 40_000.0)
    Omega = -20.0 * RAD * t + 0.7
    z_e = (0.30 + 0.04 * np.cos(2 * np.pi * t / 40_000.0)) * np.exp(1j * (omega_free + Omega))
    z_e = z_e + 0.03 * np.exp(1j * (G5 * RAD * t + 1.0)) + 0.007 * np.exp(1j * (G6 * RAD * t + 2.0))
    return zlk_series(t, z_e, Omega, np.full_like(t, np.radians(40.0)), a=a)


# ------------------------------------------------------------------------------------------------- rho_max


def test_rho_max_sees_a_free_circle_that_grows_in_part_of_the_record():
    # forced 0.03 at g5; free 0.015 for 70 % of the record, 0.045 for the last 30 %: the free circle encloses the
    # origin there, |z_free|/|z_forced| = 0.045/0.03 = 1.5, while the median free radius over the forced one says 0.5
    t = np.arange(0.0, 500_001.0, 100.0)
    radius = np.where(t < 0.7 * t[-1], 0.015, 0.045)
    z_e = 0.03 * np.exp(1j * G5 * RAD * t) + radius * np.exp(1j * -30.0 * RAD * t)
    z_i = 0.02 * np.exp(1j * PLANETARY_FREQUENCIES['s6'] * RAD * t) + 0.3 * np.exp(1j * -40.0 * RAD * t)
    Omega = np.angle(z_i)
    elements = fe.free_elements(t, np.abs(z_e), 2 * np.arcsin(np.abs(z_i)), Omega, np.angle(z_e) - Omega)
    assert elements.rho_envelope()[1] == pytest.approx(0.5, rel=0.1)  # the descriptive global ratio
    assert elements.rho_local_max(10_000.0, SETTINGS.zlk_rho_window_cycles) == pytest.approx(1.5, rel=0.1)


def test_rho_max_of_a_steady_carousel_stays_the_ratio_of_the_radii():
    # the forced radius beats between 0.045 - 0.017 and 0.045 + 0.017; the free one is 0.011: the largest ratio over a
    # window much shorter than the 54-kyr beat is 0.011 / (0.045 - 0.017) = 0.39 (the gate's 6-kyr window)
    series, elements = double_scale()
    assert elements.rho_local_max(5952.0, 1.0) == pytest.approx(0.011 / 0.028, rel=0.1)


# ------------------------------------------------------------------------------------------- co-orbital guard


def test_the_coorbital_region_of_a_planet_whose_modes_are_not_fitted():
    _, elements = genuine_librator()
    # Earth: 3 x (m_E / 3 M_sun)^(1/3) = 3 x (3.003e-6 / 3)^(1/3) = 0.0300
    assert omega_free_gate.coorbital_planet(1.0, elements, 3.0) == 'Earth'
    assert omega_free_gate.coorbital_planet(1.029, elements, 3.0) == 'Earth'
    assert omega_free_gate.coorbital_planet(1.031, elements, 3.0) == ''
    # a Jupiter Trojan: g5, g6 are fitted, the forced vector is held (591986 stays kinematic)
    assert omega_free_gate.coorbital_planet(5.2, elements, 3.0) == ''
    assert omega_free_gate.coorbital_planet(1.0, None, 3.0) == ''


def test_an_earth_coorbital_libration_has_no_gate_verdict():
    series, elements = genuine_librator(a=1.0)
    analysis = analyse_zlk(series, SETTINGS, free_elements=elements)
    assert analysis.gate.outcome == 'unavailable' and analysis.gate.detail == omega_free_gate.COORBITAL_UNFITTED
    assert (analysis.verdict.status, analysis.verdict.reason) == (Status.UNCERTAIN, Reason.GATE_UNAVAILABLE)
    # the same orbit away from any planet: a confirmed libration
    series, elements = genuine_librator(a=2.5)
    assert analyse_zlk(series, SETTINGS, free_elements=elements).verdict.status == Status.RESONANT


def test_an_episodic_coorbital_keeps_its_status():
    # 2020 PP1 (Earth co-orbital, episodic): unavailable never changes an episodic row
    result = GateResult('unavailable', n_cycles=10.0, min_cycles=3.0, rho_max=7.0, detail=omega_free_gate.COORBITAL_UNFITTED)
    decision = decide.decide_status(
        ResonanceType.LIDOV_KOZAI, decide.AngleDecision(AngleStatus.EPISODIC), result, Ledger.from_settings(SETTINGS)
    )
    assert decision.status == Status.TRANSIENT and decision.checks['check_zlk_free_omega'] == CheckValue.NOT_MEASURED


# --------------------------------------------------------------------------------- the double-scale synthetic


def test_a_forced_beat_never_becomes_the_libration_period():
    # the centre of omega swings at the beat 1296000 / (g6 - g5) = 54 027 yr; the osculating omega librates with the
    # free-forced beat 1296000 / (222 - g5) = 5952 yr. The gate's period must be the second.
    series, elements = double_scale()
    analysis = analyse_zlk(series, SETTINGS, free_elements=elements)
    row = analysis.summary_row()
    assert analysis.scale == 'fine' and row['sec_centre_rejected'] == 'forced_beat:g6-g5'
    rayleigh = AS / 500_000.0
    assert abs(AS / row['zlk_gate_period_yr'] - AS / 5952.0) < rayleigh
    assert (analysis.verdict.status, analysis.verdict.reason) == (Status.NON_RESONANT, Reason.KINEMATIC_LIBRATION)


def test_the_slow_line_of_the_centre_is_the_forced_beat():
    series, elements = double_scale()
    row = analyse_zlk(series, SETTINGS, free_elements=elements).summary_row()
    assert row['zlk_centre_origin'] == 'forced_beat:g6-g5'
    assert abs(AS / row['zlk_centre_period_yr'] - (G6 - G5)) < AS / 500_000.0
    swing = np.degrees(np.arcsin(0.017 / 0.045))  # 22.2 deg: the forced direction's half-excursion
    assert row['zlk_centre_model_half_amp_deg'] == pytest.approx(swing, rel=0.02)
    assert row['zlk_centre_half_amp_deg'] == pytest.approx(swing, rel=0.1)
    assert row['zlk_centre_model_r'] >= SETTINGS.zlk_centre_model_r_min


def test_a_kinematic_libration_is_attributed_to_its_forced_mode():
    series, elements = double_scale()
    row = analyse_zlk(series, SETTINGS, free_elements=elements).summary_row()
    assert row['zlk_attribution'] == 'attributed'
    assert row['zlk_forcing_mode'].startswith('g5')
    assert row['zlk_forcing_e'] == pytest.approx(0.045, rel=0.02)
    assert row['zlk_free_over_forcing'] == pytest.approx(0.011 / 0.045, rel=0.03)
    assert row['zlk_beat_period_yr'] == pytest.approx(AS / (222.0 - G5), rel=0.01)
    assert abs(row['zlk_node_minus_mode_turns']) < 1.0


def test_a_node_off_the_mode_is_not_attributed():
    # the node 3 Rayleigh widths off g5: the centre varpi_5 - Omega turns three times, omega circulates
    series, elements = double_scale(node_offset_widths=3.0)
    analysis = analyse_zlk(series, SETTINGS, free_elements=elements)
    assert analysis.verdict.angle_status == AngleStatus.CIRCULATION
    assert analysis.summary_row()['zlk_attribution'] == 'not_attributed'


def test_a_genuine_librator_keeps_one_timescale():
    series, elements = genuine_librator()
    analysis = analyse_zlk(series, SETTINGS, free_elements=elements)
    row = analysis.summary_row()
    assert analysis.verdict.status == Status.RESONANT and analysis.scale == 'fine'
    assert row['zlk_gate_period_yr'] == pytest.approx(40_000.0, rel=0.03)
    assert row['zlk_attribution_reason'] == 'gate_not_kinematic' and row['sec_centre_rejected'] == ''


# ------------------------------------------------------------------------------------------------ e-i coupling


def ei_record(e, inc_plane, t):
    """Ecliptic (inc, Omega) of an orbit whose node is the invariable plane's own: i_ecliptic = i' + I_inv exactly."""
    return inc_plane + I_INV, np.full_like(t, NODE_INV)


def test_constant_h_is_zlk_coupling_in_antiphase():
    # sqrt(1 - e^2) cos i = H: di/de = -e cot i / (1 - e^2) < 0 for a prograde orbit
    t = np.arange(0.0, 200_001.0, 50.0)
    e = 0.3 + 0.05 * np.cos(2 * np.pi * t / 20_000.0)
    h = np.sqrt(1 - 0.3**2) * np.cos(np.radians(50.0))
    inc, Omega = ei_record(e, np.arccos(h / np.sqrt(1 - e**2)), t)
    ei = diagnostics.measure_ei_coupling(t, e, inc, Omega, 20_000.0, 'libration', SETTINGS)
    assert ei.coupling == 'zlk' and ei.check == CheckValue.PASS and ei.confirmed_ei is True
    assert abs((ei.phase_deg - 180.0 + 180.0) % 360.0 - 180.0) < 2.0
    assert ei.band_r < -0.99 and ei.ratio_h == pytest.approx(1.0, rel=0.05)


def test_constant_q_is_a_forced_beat_in_phase():
    # sqrt(1 - e^2)(1 - cos i) = Q (forcing on varpi only): di/de = +e tan(i/2) / (1 - e^2) > 0
    t = np.arange(0.0, 200_001.0, 50.0)
    e = 0.05 + 0.01 * np.cos(2 * np.pi * t / 6_000.0)
    q = np.sqrt(1 - 0.05**2) * (1 - np.cos(np.radians(39.0)))
    inc, Omega = ei_record(e, np.arccos(1 - q / np.sqrt(1 - e**2)), t)
    ei = diagnostics.measure_ei_coupling(t, e, inc, Omega, 6_000.0, 'libration', SETTINGS)
    assert ei.coupling == 'forced_beat' and ei.check == CheckValue.FAIL and ei.confirmed_ei is False
    assert min(ei.phase_deg, 360.0 - ei.phase_deg) < 2.0
    assert ei.band_r > 0.99 and ei.ratio_beat == pytest.approx(1.0, rel=0.05)


def test_a_retrograde_orbit_is_read_by_its_tilt():
    # H < 0 (i about 130 deg): i rises with e, the tilt 180 - i falls: antiphase in the tilt, ZLK coupling
    t = np.arange(0.0, 200_001.0, 50.0)
    e = 0.3 + 0.05 * np.cos(2 * np.pi * t / 20_000.0)
    h = np.sqrt(1 - 0.3**2) * np.cos(np.radians(130.0))
    inc, Omega = ei_record(e, np.arccos(h / np.sqrt(1 - e**2)), t)
    ei = diagnostics.measure_ei_coupling(t, e, inc, Omega, 20_000.0, 'libration', SETTINGS)
    assert ei.retrograde and ei.coupling == 'zlk' and ei.ratio_h == pytest.approx(1.0, rel=0.05)


def test_an_orbit_flip_and_a_short_record_are_not_measured():
    t = np.arange(0.0, 200_001.0, 50.0)
    e = np.full_like(t, 0.3)
    inc, Omega = ei_record(e, np.radians(90.0) + np.radians(5.0) * np.sin(2 * np.pi * t / 20_000.0), t)
    assert diagnostics.measure_ei_coupling(t, e, inc, Omega, 20_000.0, 'libration', SETTINGS).reason == 'orbit_flip'
    inc, Omega = ei_record(e, np.full_like(t, 0.5), t)
    short = diagnostics.measure_ei_coupling(t, e, inc, Omega, 50_000.0, 'libration', SETTINGS)
    assert (short.coupling, short.reason, short.check) == ('not_measured', 'short_record', CheckValue.NOT_MEASURED)


def test_independent_lines_are_no_coupling():
    t = np.arange(0.0, 200_001.0, 50.0)
    e = 0.3 + 0.05 * np.cos(2 * np.pi * t / 20_000.0)
    inc, Omega = ei_record(e, 0.6 + 0.02 * np.cos(2 * np.pi * t / 7_300.0), t)
    assert diagnostics.measure_ei_coupling(t, e, inc, Omega, 20_000.0, 'libration', SETTINGS).coupling == 'none'


def test_a_circulating_omega_is_read_at_its_quadrupole_harmonic():
    # omega turns once in 4 kyr: e and i trade at cos 2 omega, every 2 kyr (15527: 2.05 kyr)
    t = np.arange(0.0, 50_001.0, 10.0)
    period, source = diagnostics.decisive_period(AngleStatus.CIRCULATION, float('nan'), t, 2 * np.pi * t / 4000.0)
    assert (period, source) == (pytest.approx(2000.0), 'circulation_harmonic')


# --------------------------------------------------------------------------------------- real fixtures (500 kyr)


def load_fixture(name):
    config = SimulationConfig(save=None, plot=None, save_summary=False, integration_years=5e5)
    with gzip.open(FIXTURES / f'{name}.csv.gz', 'rt') as handle:
        data = pd.read_csv(handle)
    series = LongPeriodSeries(
        ResonanceType.LIDOV_KOZAI,
        'LK',
        name,
        data['times'].values,
        np.unwrap(data['LK_angle_unwrapped'].values),
        data['a'].values,
        data['e'].values,
        data['inc'].values,
        varpi=data['Omega'].values + data['omega'].values,
        Omega=data['Omega'].values,
    )
    elements = fe.free_elements(
        data['times'].values,
        data['e'].values,
        data['inc'].values,
        data['Omega'].values,
        data['omega'].values,
        sampling_years=config.free_elements_sampling_years,
    )
    return analyse_zlk(series, settings_from_config(config), free_elements=elements, gate_config=config)


def test_591986_the_trojan_beat_and_its_forcing():
    analysis = load_fixture('591986')
    row = analysis.summary_row()
    assert (analysis.verdict.status, analysis.verdict.reason) == (Status.NON_RESONANT, Reason.KINEMATIC_LIBRATION)
    assert row['zlk_gate'] == 'kinematic' and row['zlk_gate_detail'] == '' and row['zlk_gate_rho_max'] < 1.0
    # the 6-kyr cycle: i in phase with e at about the forced-beat law (studies: 329 deg, 0.0041 deg vs 0.0037)
    assert row['zlk_ei_coupling'] == 'forced_beat'
    assert min(row['zlk_ei_phase_deg'], 360.0 - row['zlk_ei_phase_deg']) < 45.0
    assert 0.5 < row['zlk_ei_ratio_beat'] < 2.0
    # forced by g5 (with its cluster), free/forcing 0.25, the beat 2 pi/|b_varpi - g5| = P_cyc
    assert row['zlk_attribution'] == 'attributed' and row['zlk_forcing_mode'].startswith('g5')
    assert row['zlk_free_over_forcing'] == pytest.approx(0.25, abs=0.03)
    assert abs(AS / row['zlk_beat_period_yr'] - AS / row['zlk_gate_period_yr']) < AS / 5e5
    # the centre: g6-g5 at 54.0 kyr, +-21.6 deg against arcsin(|c6|/|c5|) = 22.8 deg, reconstruction r 0.99
    assert row['zlk_centre_origin'] == 'forced_beat:g6-g5'
    assert row['zlk_centre_period_yr'] == pytest.approx(54_027.0, abs=5_800.0)  # a Rayleigh width P^2 / T
    assert row['zlk_centre_half_amp_deg'] == pytest.approx(21.6, abs=1.5)
    assert row['zlk_centre_model_half_amp_deg'] == pytest.approx(22.8, abs=0.5)
    assert row['zlk_centre_model_r'] > 0.98


@pytest.mark.parametrize('name', ['3040', '162474', '4257'])
def test_librators_show_zlk_coupling_at_their_period(name):
    analysis = load_fixture(name)
    row = analysis.summary_row()
    assert analysis.verdict.status == Status.RESONANT
    assert row['zlk_ei_coupling'] == 'zlk' and row['confirmed_ei'] is True
    assert row['zlk_ei_period_yr'] == pytest.approx(row['zlk_gate_period_yr'])
    assert 175.0 < row['zlk_ei_phase_deg'] < 185.0 and row['zlk_ei_band_r'] < -0.99
    assert analysis.verdict.checks['check_zlk_ei'] == CheckValue.PASS


# ------------------------------------------------------------------------------- persistence and the plane


def test_mode_amplitudes_survive_omega_free_csv(tmp_path):
    _, elements = double_scale()
    path = tmp_path / 'x-omega-free.csv'
    pd.DataFrame(fe.series_rows(elements)).to_csv(path, index=False)
    setup = {'basis_e': list(elements.basis_e), 'basis_i': list(elements.basis_i), 'clusters_e': elements.clusters_e}
    restored = fe.from_series_frame(pd.read_csv(path), setup)
    for label, value in elements.amplitudes_e.items():
        assert restored.amplitudes_e[label] == pytest.approx(value, rel=1e-9)
    np.testing.assert_allclose(restored.z_e_forced, elements.z_e_forced, rtol=1e-6, atol=1e-12)
    # a file written before the amplitudes: magnitudes only, and no attribution
    old = pd.read_csv(path)
    old = old[[column for column in old.columns if not column.startswith(fe.AMPLITUDE_PREFIX)]]
    legacy = fe.from_series_frame(old, setup)
    assert legacy.amplitudes_e == {}
    assert diagnostics.attribute_forcing(legacy, 'kinematic', 5952.0, 0.4).reason == 'no_mode_amplitudes'


def _ecliptic_elements(inc_p, node_p, omega_p):
    """Ecliptic (inc, Omega, omega) of an orbit given in the invariable frame, by the inverse rotation R^T."""
    h = np.array([np.sin(inc_p) * np.sin(node_p), -np.sin(inc_p) * np.cos(node_p), np.cos(inc_p)])
    node_vec = np.array([np.cos(node_p), np.sin(node_p), 0.0])
    p = np.cos(omega_p) * node_vec + np.sin(omega_p) * np.cross(h, node_vec)
    rotation = rotation_to_invariable().T
    h, p = rotation @ h, rotation @ p
    inc = np.arccos(h[2])
    node = np.arctan2(h[0], -h[1])
    line = np.array([np.cos(node), np.sin(node), 0.0])
    return inc, node, np.arctan2(p @ np.cross(h, line), p @ line)


def test_lidov_integrals_are_read_in_the_invariable_plane():
    # e = 0.4, i' = 60 deg, omega' = 90 deg: c1 = 0.84 x 0.25 = 0.21, c2 = 0.16 (0.4 - 0.75) = -0.056,
    # h_quad = 6 c1 - 2 + 30 c2 = -2.42 (Codex review, item 4)
    inc, node, omega = _ecliptic_elements(np.radians(60.0), np.radians(30.0), np.radians(90.0))
    c1, c2, h_quad = LidovKozaiParameters.evaluate_invariable(0.4, inc, node, omega)
    assert (float(c1), float(c2), float(h_quad)) == (pytest.approx(0.21), pytest.approx(-0.056), pytest.approx(-2.42))
    # an orbit in the invariable plane: c1 = 1 - e^2 exactly, c2 = 2/5 e^2
    c1, c2, _ = LidovKozaiParameters.evaluate_invariable(0.3, I_INV, NODE_INV, 1.2)
    assert float(c1) == pytest.approx(1 - 0.09, abs=1e-12) and float(c2) == pytest.approx(0.4 * 0.09, abs=1e-12)
    assert inclination_to_invariable_plane(I_INV, NODE_INV) == pytest.approx(0.0, abs=1e-7)


# ------------------------------------------------------------------------------------------------------ figure


def test_the_zlk_figure_draws_the_free_omega_in_turns_and_names_the_measured_facts():
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from resonances.plotting.secular_plots import draw_long_period_combined

    series, elements = double_scale()
    analysis = analyse_zlk(series, SETTINGS, free_elements=elements)
    fig = draw_long_period_combined(analysis, free_elements=elements, settings=SETTINGS)
    try:
        panel = next(ax for ax in fig.axes if ax.get_ylabel() == 'ω drift, turns')
        assert [text.get_text() for text in panel.get_legend().get_texts()] == ['osculating ω', 'free ω']
        # the free omega winds (222 - g5) x 500 kyr / 1296000 = 84 turns; the panel's top holds them
        assert panel.get_ylim()[1] > 80
        info = ' '.join(fig.text_stats['text'].info)
        assert 'forced beat g6-g5' in info and 'held by the forced mode g5' in info and 'e-i at' in info
        assert 'cycles of the free' not in ' '.join(fig.text_stats['text'].sentences)
    finally:
        plt.close(fig)


# ----------------------------------------------------------------------------------- modulated secular angle


def test_the_slow_line_of_a_modulated_secular_centre_is_named_after_its_forced_wave():
    from resonances.resonance.analysis.secular import slow
    from tests.resonances.analysis.builders import make_centre

    # a centre swinging +-10 deg at 1.2 Myr over 10 Myr, smoothed at W_c = 300 kyr; the angle's forced waves: g6 at
    # 1.25 Myr (|1/1.2 - 1/1.25| Myr^-1 = 0.033 < 1/T = 0.1: one line) and g5 at 3 Myr
    t = np.arange(0.0, 10e6 + 1.0, 1000.0)
    track = make_centre(t_yr=t, centre_rad=np.pi + np.radians(10.0) * np.sin(2 * np.pi * t / 1.2e6), window_yr=3e5)
    line = slow.secular_centre_line(track, {'g6': 1.25e6, 'g5': 3e6}, 10e6, SETTINGS)
    assert line.period_yr == pytest.approx(1.2e6, rel=0.02)
    assert np.degrees(line.half_amp_rad) == pytest.approx(10.0, rel=0.02)
    assert line.origin == 'forced_wave:g6'
    assert slow.secular_centre_line(track, {'g5': 3e6}, 10e6, SETTINGS).origin == 'unexplained'


# ------------------------------------------------------------------- W25 independent review (Codex), findings 1, 2, 4, 5


def test_a_disabled_gate_keeps_the_geometric_result_with_or_without_the_split():
    # finding 1: the centre-scale guard reads the forced/free split, so it acts only where the gate does. With the gate
    # disabled the double-scale synthetic decides at its centre scale, the g6-g5 beat 1296000 / (g6 - g5) = 54.0 kyr,
    # whether the split is supplied or not; the slow line of the centre is still named (a column)
    series, elements = double_scale()
    disabled = SimulationConfig(save=None, plot=None, save_summary=False, free_gate_enabled=False)
    with_split = analyse_zlk(series, SETTINGS, free_elements=elements, gate_config=disabled)
    without = analyse_zlk(series, SETTINGS, free_elements=None, gate_config=disabled)
    for analysis in (with_split, without):
        assert analysis.gate.outcome == 'disabled' and analysis.centre_rejected == '' and analysis.scale == 'centre'
        assert abs(AS / analysis.geometry.period_cycle_yr - (G6 - G5)) < AS / 500_000.0
    keys = ('status', 'angle_status', 'reason')
    assert [str(getattr(with_split.verdict, k)) for k in keys] == [str(getattr(without.verdict, k)) for k in keys]
    assert with_split.summary_row()['zlk_centre_origin'] == 'forced_beat:g6-g5'


def test_a_coorbital_with_unfitted_forcing_has_no_centre_guard():
    # the same orbit at a = 1 au, inside Earth's co-orbital region (g3 is not fitted): the gate has no verdict, and the
    # split's beats refuse nothing either
    series, elements = double_scale(a=1.0)
    analysis = analyse_zlk(series, SETTINGS, free_elements=elements)
    assert analysis.gate.detail == omega_free_gate.COORBITAL_UNFITTED and analysis.centre_rejected == ''
    assert analysis.scale == 'centre' and analysis.verdict.reason == Reason.GATE_UNAVAILABLE


def _with_forcing(free_radius, amplitudes):
    """The double-scale split with the forced vector replaced by `amplitudes` (complex, at their g's) and the free
    vector by a circle of radius `free_radius(t)` at 222"/yr; the free node at g5."""
    from dataclasses import replace

    _, elements = double_scale()
    t = elements.times
    forced = sum(value * np.exp(1j * PLANETARY_FREQUENCIES[label] * RAD * t) for label, value in amplitudes.items())
    free = free_radius(t) * np.exp(1j * 222.0 * RAD * t)
    basis = {label: PLANETARY_FREQUENCIES[label] for label in amplitudes}
    return replace(
        elements,
        z_e=forced + free,
        z_e_free=free,
        z_e_forced=forced,
        basis_e=basis,
        amplitudes_e=dict(amplitudes),
        b_varpi=222.0,
        b_Omega=G5,
        clusters_e=[[label] for label in amplitudes],
    )


def test_the_strict_pin_bounds_the_free_radius_over_the_whole_record():
    # finding 2: forced 0.05 at g5; free 0.01 for 70 % of 500 kyr and 0.10 for the last 30 %. Over the last 150 kyr
    # |z_free| = 0.10 > |z_forced| = 0.05: the free circle encloses the origin (local ratio 2), although the median
    # free radius 0.01 is a fifth of the forcing. Not attributed.
    period = AS / (222.0 - G5)
    growing = _with_forcing(lambda t: np.where(t < 0.7 * t[-1], 0.01, 0.10), {'g5': 0.05 + 0j})
    rho = growing.rho_local_max(period, SETTINGS.zlk_rho_window_cycles)
    assert rho == pytest.approx(0.10 / 0.05, rel=0.02) and growing.e_free == pytest.approx(0.01)
    result = diagnostics.attribute_forcing(growing, 'kinematic', period, rho)
    assert (result.attributed, result.reason) == (False, 'free_circle_reaches_origin')
    # the control: a free circle of 0.01 throughout is pinned strictly (0.01 + 0 < 0.05) and attributed
    steady = _with_forcing(lambda t: np.full_like(t, 0.01), {'g5': 0.05 + 0j})
    result = diagnostics.attribute_forcing(steady, 'kinematic', period, steady.rho_local_max(period, 1.0))
    assert result.attributed and result.strict_pin is True and result.mode == 'g5'


def test_an_undersampled_e_i_band_is_not_measured():
    # finding 4: e = 0.3 + 0.01 cos(2 pi t / 150 yr) with sqrt(1 - e^2) cos i constant, sampled every 100 yr: the line
    # at 150 yr is above the Nyquist frequency 1 / 200 yr, so the sampling can neither confirm nor reject a law
    period = 150.0
    coarse = np.arange(0.0, 100_001.0, 100.0)
    e = 0.3 + 0.01 * np.cos(2 * np.pi * coarse / period)
    inc, Omega = ei_record(e, np.arccos(np.sqrt(1 - 0.09) * np.cos(0.6) / np.sqrt(1 - e**2)), coarse)
    ei = diagnostics.measure_ei_coupling(coarse, e, inc, Omega, period, 'circulation_harmonic', SETTINGS)
    assert (ei.coupling, ei.reason, ei.check, ei.confirmed_ei) == ('not_measured', 'undersampled', CheckValue.NOT_MEASURED, None)
    # the same orbit sampled every 10 yr: ZLK coupling, antiphase at the constant-H amplitude
    fine = np.arange(0.0, 20_001.0, 10.0)
    e = 0.3 + 0.01 * np.cos(2 * np.pi * fine / period)
    inc, Omega = ei_record(e, np.arccos(np.sqrt(1 - 0.09) * np.cos(0.6) / np.sqrt(1 - e**2)), fine)
    resolved = diagnostics.measure_ei_coupling(fine, e, inc, Omega, period, 'circulation_harmonic', SETTINGS)
    assert resolved.coupling == 'zlk' and resolved.ratio_h == pytest.approx(1.0, rel=0.05)


def _beat_track(half_amp_deg, beat):
    """A centre swinging +-half_amp_deg at the beat period (W_c 6 kyr), sampled like the double-scale record."""
    from tests.resonances.analysis.builders import make_centre

    t = np.arange(0.0, 500_001.0, 100.0)
    return t, make_centre(t_yr=t, centre_rad=np.radians(half_amp_deg) * np.sin(2 * np.pi * t / beat), window_yr=6000.0)


def test_a_beat_refuses_the_centre_scale_only_when_the_forced_terms_make_that_wave():
    # finding 5: a centre swinging +-8 deg at 1296000 / (g6 - g5) = 54.0 kyr (a wave of the node, say). The phase wave
    # of c5 + c6 exp(i (g6 - g5) t) swings +-arcsin(|c6| / |c5|): it is this wave only when |c6| = |c5| sin 8 deg.
    beat = AS / (G6 - G5)
    t, track = _beat_track(8.0, beat)
    c5 = 0.05 + 0j
    explaining = _with_forcing(lambda x: np.full_like(x, 0.01), {'g5': c5, 'g6': c5 * np.sin(np.radians(8.0))})
    assert diagnostics.forced_beat_centre(explaining, beat, 0.2, track, t, SETTINGS) == 'forced_beat:g6-g5'
    # a zero g6 term: the forced vector has constant length and turns uniformly, no beat at all (reviewer's input)
    zero = _with_forcing(lambda x: np.full_like(x, 0.01), {'g5': c5, 'g6': 0j})
    assert np.ptp(np.abs(zero.forced_vector(t))) < 1e-12 and zero.forced_beats() == {}
    assert diagnostics.forced_beat_centre(zero, beat, 0.2, track, t, SETTINGS) == ''
    # a g6 term a hundred times too small: the same shape (r = 1.000), a hundredth of the swing (gain 0.01): not this
    # wave; correlation alone is blind to size
    tiny = _with_forcing(lambda x: np.full_like(x, 0.01), {'g5': c5, 'g6': c5 * np.sin(np.radians(8.0)) / 100.0})
    assert diagnostics.forced_beat_centre(tiny, beat, 0.2, track, t, SETTINGS) == ''
    # an old file: no per-mode amplitudes, no measured beat
    from dataclasses import replace

    legacy = replace(explaining, amplitudes_e={})
    assert legacy.forced_beats() == {} and diagnostics.forced_beat_centre(legacy, beat, 0.2, track, t, SETTINGS) == ''
