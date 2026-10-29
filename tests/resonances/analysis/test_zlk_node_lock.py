"""The node lock of a ZLK row, the attribution inside episodes, the cause sentence and the forcing panel (a′).

Expectations are computed by hand from the synthetic vectors, not from the package: a forced eccentricity 0.045 at g5
with phase 83 deg plus a free circle; the node Omega = g5 t + 146 deg + A sin(2 pi t / P_lock). Then
Omega - varpi5 = 63 deg + A sin(2 pi t / P_lock): its range is 63 -/+ A over full cycles, and its turning points sit at
t = (k + 1/2) P_lock / 2. The smoothing of the node difference is its longest forced wave, the constant inclination
term at 1296000 / |b_Omega| ~ 306 kyr (b_Omega = g5 = 4.2575"/yr), trimmed once at each edge (FILTER_EDGE_TRIM_CUTOFFS
= 1). 591986 over 4 Myr (dev notes): Omega - varpi5 within 52-129 deg, a libration of about 1.46 Myr.
"""

from dataclasses import replace

import numpy as np
import pytest

from resonances.data.const import PLANETARY_FREQUENCIES
from resonances.plotting.style import LAYOUT_V1
from resonances.resonance.analysis.secular import LongPeriodSeries
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import ResonanceType
from resonances.resonance.analysis.zlk import analyse_zlk, diagnostics
from resonances.secular import free_elements as fe

SETTINGS = settings_from_config()
AS = 1296000.0
RAD = 2 * np.pi / AS  # rad/yr per arcsec/yr
G5, G6 = PLANETARY_FREQUENCIES['g5'], PLANETARY_FREQUENCIES['g6']


def zlk_record(t, z_e, Omega, inc_deg=39.0):
    """(LongPeriodSeries of omega = arg z_e - Omega, its FreeElements)."""
    e, varpi = np.abs(z_e), np.angle(z_e)
    inc = np.full_like(t, np.radians(inc_deg))
    omega = np.unwrap(varpi - Omega)
    series = LongPeriodSeries(
        ResonanceType.LIDOV_KOZAI, 'LK', 'synthetic', t, omega, np.full_like(t, 5.2), e, inc, varpi=np.unwrap(varpi), Omega=Omega
    )
    return series, fe.free_elements(t, e, inc, Omega, omega)


def locked_4myr(node_amplitude_deg=40.0, node_offset_widths=0.0):
    """4 Myr: forced g5 only; the free circle at g5 + 216"/yr, so the beat is 1296000 / 216 = 6000 yr exactly (on the
    sampling grid: P_cyc resolves it within the 4-Myr Rayleigh width); the node librates about varpi5 + 63 deg with
    `node_amplitude_deg` at 1.4 Myr (+ an offset of its rate in Rayleigh widths of 4 Myr)."""
    t = np.arange(0.0, 4_000_001.0, 100.0)
    z_e = 0.045 * np.exp(1j * (G5 * RAD * t + np.radians(83.0))) + 0.011 * np.exp(1j * (G5 + 216.0) * RAD * t)
    rate = G5 + node_offset_widths * AS / 4_000_000.0
    Omega = rate * RAD * t + np.radians(146.0) + np.radians(node_amplitude_deg) * np.sin(2 * np.pi * t / 1.4e6)
    return zlk_record(t, z_e, Omega)


def trojan_500kyr():
    """500 kyr, 591986's analogue: forced 0.045 at g5 + 0.017 at g6, the free circle 0.011 at 222"/yr, the node at g5
    with an 8-deg wave of 1.5 Myr (a third of a cycle fits)."""
    t = np.arange(0.0, 500_001.0, 100.0)
    z_e = 0.045 * np.exp(1j * (G5 * RAD * t + np.radians(83.0))) + 0.017 * np.exp(1j * (G6 * RAD * t + np.radians(3.2)))
    z_e = z_e + 0.011 * np.exp(1j * 222.0 * RAD * t)
    Omega = G5 * RAD * t + np.radians(146.0) + np.radians(8.0) * np.sin(2 * np.pi * t / 1.5e6)
    return zlk_record(t, z_e, Omega)


def genuine_librator():
    """A ZLK librator: free omega = 90 + 25 sin(2 pi t / 40 kyr), e_free 0.30, its own node at -20"/yr, small forced
    terms; the gate confirms it, nothing induces it."""
    t = np.arange(0.0, 500_001.0, 100.0)
    omega_free = np.radians(90.0) + np.radians(25.0) * np.sin(2 * np.pi * t / 40_000.0)
    Omega = -20.0 * RAD * t + 0.7
    z_e = (0.30 + 0.04 * np.cos(2 * np.pi * t / 40_000.0)) * np.exp(1j * (omega_free + Omega))
    z_e = z_e + 0.03 * np.exp(1j * (G5 * RAD * t + 1.0)) + 0.007 * np.exp(1j * (G6 * RAD * t + 2.0))
    return zlk_record(t, z_e, Omega, inc_deg=40.0)


# --------------------------------------------------------------------------------------------------- node lock


def test_a_node_that_librates_about_g5_is_locked():
    _, elements = locked_4myr()
    lock = diagnostics.node_lock(elements, SETTINGS)
    assert (lock.mode, lock.reason, lock.candidate) == ('g5', '', 'g5')
    # 63 -/+ 40 deg; the extrema at 0.35, 1.05, 1.75, 2.45, 3.15 Myr lie inside [306 kyr, 3.69 Myr] (3.85 does not):
    # five turning points, two full cycles, period 2 x (3.15 - 0.35) / 4 = 1.4 Myr
    assert lock.min_deg == pytest.approx(23.0, abs=0.5) and lock.max_deg == pytest.approx(103.0, abs=0.5)
    assert lock.cycles == 2.0 and lock.period_yr == pytest.approx(1.4e6, rel=0.02)
    assert abs(lock.rate_diff_arcsec_yr) < lock.resolution_arcsec_yr == pytest.approx(AS / 4e6)
    assert lock.to_columns()['zlk_node_lock'] == 'g5'


def test_a_node_off_the_mode_or_unbounded_is_not_locked():
    # three Rayleigh widths off g5: Omega - varpi5 turns three times over the record
    _, elements = locked_4myr(node_offset_widths=3.0)
    lock = diagnostics.node_lock(elements, SETTINGS)
    assert (lock.mode, lock.reason) == ('', 'rate_mismatch')
    assert lock.rate_diff_arcsec_yr == pytest.approx(3 * AS / 4e6, rel=0.1)
    # the rate within the resolution but a swing of +-200 deg: 400 deg, more than a turn
    _, elements = locked_4myr(node_amplitude_deg=200.0)
    lock = diagnostics.node_lock(elements, SETTINGS)
    assert (lock.mode, lock.reason) == ('', 'unbounded')
    assert lock.max_deg - lock.min_deg == pytest.approx(400.0, abs=2.0)


def test_a_lock_the_record_cannot_resolve():
    # 500 kyr of a 1.5-Myr wave: 63 + 8 sin(2 pi t / 1.5 Myr) rises from 63 to its top 71 at 375 kyr and falls back to
    # 69.9 deg: the rate matches g5 and the difference stays within 8 deg, but no cycle closes (smoothed at ~306 kyr
    # and trimmed by it at each edge, nothing is left to turn)
    _, elements = trojan_500kyr()
    lock = diagnostics.node_lock(elements, SETTINGS)
    assert (lock.mode, lock.reason, lock.candidate) == ('', 'unresolved', 'g5')
    assert lock.min_deg == pytest.approx(63.0, abs=0.5) and lock.max_deg == pytest.approx(71.0, abs=0.5)
    assert lock.cycles < SETTINGS.zlk_node_lock_min_cycles and not np.isfinite(lock.period_yr)


def test_a_lock_needs_the_mode_amplitudes():
    _, elements = trojan_500kyr()
    assert diagnostics.node_lock(replace(elements, amplitudes_e={}), SETTINGS).reason == 'no_mode_amplitudes'
    assert diagnostics.node_lock(None, SETTINGS).reason == 'no_mode_amplitudes'


# ----------------------------------------------------------------------------------------- episodic attribution


def _growing_free_circle(node_rate):
    """Forced 0.05 at g5; the free circle 0.01 for 70 % of 500 kyr, 0.10 after (rho 2 there: the origin is enclosed
    over the record); the free node at `node_rate`."""
    _, elements = trojan_500kyr()
    t = elements.times
    forced = 0.05 * np.exp(1j * G5 * RAD * t)
    free = np.where(t < 0.7 * t[-1], 0.01, 0.10) * np.exp(1j * 222.0 * RAD * t)
    return replace(
        elements,
        z_e=forced + free,
        z_e_free=free,
        z_e_forced=forced,
        basis_e={'g5': G5},
        amplitudes_e={'g5': 0.05 + 0j},
        b_varpi=222.0,
        b_Omega=node_rate,
        clusters_e=[['g5']],
    )


def test_an_episodic_row_pinned_in_its_episodes_is_attributed_there():
    period = AS / (222.0 - G5)  # 5952 yr: the beat the osculating omega librates with
    elements = _growing_free_circle(G5 - 17.2)  # the node 17.2"/yr off g5: 6.6 turns of varpi5 - Omega in 500 kyr
    # pinned inside the episodes (median local rho 0.95 < 1, one turn of the free omega per cycle >= 0.5)
    result = diagnostics.attribute_forcing(elements, 'kinematic', period, 2.0, 0.95, 1.0, 0.5)
    assert (result.attributed, result.scope, result.mode) == (True, 'episodes', 'g5')
    assert result.node_minus_mode_turns == pytest.approx(17.2 * 500_000 / AS, rel=0.01)  # 6.64: reported, no veto
    # the free omega librating inside the episodes (0.2 turns per cycle) is not the decision's pin: not attributed
    result = diagnostics.attribute_forcing(elements, 'kinematic', period, 2.0, 0.95, 0.2, 0.5)
    assert (result.attributed, result.reason) == (False, 'free_circle_reaches_origin')
    # pinned over the whole record, the same node is vetoed: the centre would turn under a whole-record libration
    result = diagnostics.attribute_forcing(elements, 'kinematic', period, 0.5)
    assert (result.attributed, result.reason) == (False, 'node_does_not_follow_mode')


# --------------------------------------------------------------------------------------------- the cause sentence


def test_the_cause_sentence_names_the_mode_the_beat_and_the_lock():
    from resonances.plotting.verdict_text import zlk_cause, zlk_column_lines

    series, elements = locked_4myr()
    analysis = analyse_zlk(series, SETTINGS, free_elements=elements)
    sentence, used = zlk_cause(analysis)
    assert sentence.startswith('Induced by the forced eccentricity of mode g5 (|c| 0.045), which holds ϖ')
    assert 'the beat 2π/|b_ϖ − g5| 6 kyr = P_cyc' in sentence  # 1296000 / 216 = 6000 yr
    assert 'The node is locked to g5: Ω − ϖ5 within 23–103° (2 cycles of 1.4 Myr).' in sentence
    assert set(used) >= {'attribution', 'node_lock'}
    # what the sentence says is not repeated in the grey line
    assert not any('locked' in line or 'held by the forced mode' in line for line in zlk_column_lines(analysis))


def test_the_cause_sentence_says_when_a_lock_is_not_resolved():
    from resonances.plotting.verdict_text import zlk_cause

    series, elements = trojan_500kyr()
    sentence = zlk_cause(analyse_zlk(series, SETTINGS, free_elements=elements))[0]
    assert 'The centre\'s' in sentence and 'is the g6−g5 beat of the forced vector' in sentence
    assert 'The node follows g5 within the resolution' in sentence and 'a lock is not resolved' in sentence


def test_no_cause_sentence_for_a_confirmed_libration():
    from resonances.plotting.verdict_text import zlk_cause

    series, elements = genuine_librator()
    analysis = analyse_zlk(series, SETTINGS, free_elements=elements)
    assert analysis.gate.outcome == 'confirmed'
    assert zlk_cause(analysis) == ('', ())


# ------------------------------------------------------------------------------------------------- panel (a′)


def _figure(series, elements):
    import matplotlib

    matplotlib.use('Agg')
    from resonances.plotting.secular_plots import draw_long_period_combined

    analysis = analyse_zlk(series, SETTINGS, free_elements=elements)
    return analysis, draw_long_period_combined(analysis, free_elements=elements, settings=SETTINGS)


def _time_panels(fig):
    """The time panels: axes sharing the x axis of the first one."""
    first = fig.axes[0]
    return [ax for ax in fig.axes if ax is first or first.get_shared_x_axes().joined(first, ax)]


def test_an_induced_libration_gets_the_forcing_panel():
    import matplotlib.pyplot as plt

    series, elements = trojan_500kyr()
    _, fig = _figure(series, elements)
    try:
        panels = _time_panels(fig)
        assert len(panels) == 8
        legend = [text.get_text() for text in panels[1].get_legend().get_texts()]
        assert legend[0] == 'ω_f'
        # the two largest fitted amplitudes and their beat 1296000 / (g6 - g5) = 54.0 kyr
        assert legend[1].startswith('arg z_forced − Ω: the forced direction (|c| g5 0.045') and 'beat 54 kyr' in legend[1]
        assert legend[2].startswith('ϖ5 − Ω: the mode g5 against the node')
    finally:
        plt.close(fig)


def test_a_confirmed_libration_keeps_the_seven_panels_in_place():
    import matplotlib.pyplot as plt

    from resonances.plotting.mmr_combined import panel_rects

    series, elements = genuine_librator()
    _, fig = _figure(series, elements)
    try:
        panels = _time_panels(fig)
        assert len(panels) == 7
        width, height = fig.get_size_inches()
        for ax, (x, y, w, h) in zip(panels, panel_rects(LAYOUT_V1, 7)):
            assert ax.get_position().bounds == pytest.approx((x / width, y / height, w / width, h / height))
        assert not any(ax.get_legend() and 'arg z_forced' in ' '.join(t.get_text() for t in ax.get_legend().get_texts()) for ax in fig.axes)
    finally:
        plt.close(fig)
