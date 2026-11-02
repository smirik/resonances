"""The Lidov-Kozai angle measured from the invariable plane (W28 D2, card g09-1, author's option A).

From the ecliptic, the node of an orbit tilted from the invariable plane by less than the plane's own 1.5787 deg
librates about the plane's node instead of turning, and omega = varpi - Omega with it: 555527 (i_inv about 1.6 deg)
looked bounded (a kinematic libration) and circulates from the invariable plane. ZLK_ANGLE_FRAME = invariable (the
default) builds omega, i and Omega by an exact rotation of the orbit's vectors (`elements_in_frame`), on the analysis
path (`simulation.long_period_series`); the e-i coupling must then read the inclination as it is (no second rotation).

Independent values: the plane's tilt and node (Souami & Souchay 2012, A&A 543, A133: 1.5787 deg, 107.5822 deg); an
ecliptic orbit (i = 0) is tilted by exactly that angle from the plane, ascends on it at the plane's node + 180 deg,
and a perihelion on the line of the plane's node stays there (the rotation is about that line).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from resonances.body import Body
from resonances.data.invariable_plane import (
    FRAME_ECLIPTIC,
    FRAME_INVARIABLE,
    argument_of_pericentre,
    elements_from_invariable_plane,
    elements_in_invariable_plane,
    inclination_to_invariable_plane,
)
from resonances.lidov_kozai.lidov_kozai_resonance import LidovKozaiResonance, elements_in_frame
from resonances.resonance.analysis.rules import Rule
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, Status
from resonances.resonance.analysis.zlk import analyse_zlk, diagnostics
from resonances.simulation.simulation import _store_zlk_angle, long_period_series

SETTINGS = settings_from_config()
NO_GATE = type('NoGate', (), {'free_gate_enabled': False, 'free_gate_min_cycles': 3.0})()
I_PLANE, NODE_PLANE = np.radians(1.5787), np.radians(107.5822)  # Souami & Souchay (2012)


def turns(angle) -> float:
    unwrapped = np.unwrap(np.asarray(angle, dtype=float))
    return float((unwrapped[-1] - unwrapped[0]) / (2 * np.pi))


def wrapped_difference(a, b) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b)))))


def zlk_body(t_yr, ecc, inc, Omega, omega) -> Body:
    """A body with ecliptic elements and the ecliptic omega stored as the integration stores the ZLK angle."""
    body = Body()
    body.name = 'synthetic'
    body.times = np.asarray(t_yr, dtype=float) * 2 * np.pi
    body.axis = np.full_like(t_yr, 2.5)
    body.ecc, body.inc, body.Omega, body.omega = (np.asarray(x, dtype=float) for x in (ecc, inc, Omega, omega))
    body.varpi = np.mod(body.Omega + body.omega, 2 * np.pi)
    resonance = LidovKozaiResonance()
    body.lidov_kozai_resonances.append(resonance)
    body.angles_unwrapped[resonance.to_s()] = np.unwrap(body.omega)
    body.angles[resonance.to_s()] = np.mod(body.omega, 2 * np.pi)
    return body, resonance


def low_tilt_orbit(n=10_001, span=200_000.0, i_inv_deg=1.2):
    """555527's geometry: tilted from the invariable plane by less than the plane's own tilt, its node turning on the
    plane once per 25 kyr, its perihelion bounded (a forced-eccentricity-dominated orbit): omega circulates from the
    invariable plane (+8 turns) and is bounded from the ecliptic (the ecliptic node librates about the plane's)."""
    t = np.linspace(0.0, span, n)
    inc_inv = np.full_like(t, np.radians(i_inv_deg))
    node_inv = 0.3 - 2 * np.pi * t / 25_000.0
    varpi = 1.0 + np.radians(20.0) * np.sin(2 * np.pi * t / 7_000.0)
    inc, node, omega = elements_from_invariable_plane(inc_inv, node_inv, varpi - node_inv)
    return t, inc, node, omega


# ----------------------------------------------------------------------------------------------- the rotation


def test_an_ecliptic_orbit_seen_from_the_invariable_plane():
    # i = 0: tilted by the plane's tilt, ascending at the plane's node + 180 deg; a perihelion at longitude Omega_I
    # lies on the rotation axis and keeps its longitude, so omega' = Omega_I - (Omega_I + pi) = -pi
    node = np.radians([0.0, 40.0, 250.0])  # undefined for i = 0: any value describes the same orbit
    inc, node_new, omega_new = elements_in_invariable_plane(np.zeros(3), node, NODE_PLANE - node)
    np.testing.assert_allclose(inc, I_PLANE, atol=1e-6)
    np.testing.assert_allclose(wrapped_difference(node_new, NODE_PLANE + np.pi), 0.0, atol=1e-6)
    np.testing.assert_allclose(wrapped_difference(omega_new, np.pi), 0.0, atol=1e-6)


def test_the_round_trip_is_exact():
    rng = np.random.default_rng(7)
    inc = np.concatenate([rng.uniform(0.05, 1.5, 200), rng.uniform(1.65, 3.09, 200)])  # prograde and retrograde
    node, omega = rng.uniform(-np.pi, np.pi, 400), rng.uniform(-np.pi, np.pi, 400)
    back = elements_from_invariable_plane(*elements_in_invariable_plane(inc, node, omega))
    np.testing.assert_allclose(back[0], inc, atol=1e-12)
    assert wrapped_difference(back[1], node).max() < 1e-12
    assert wrapped_difference(back[2], omega).max() < 1e-12


def test_the_inclination_agrees_with_the_angular_momentum_formula():
    # two routes: the rotated vectors (elements_in_frame) and arccos(h . n) without any node
    rng = np.random.default_rng(11)
    inc, node, omega = rng.uniform(0.01, 3.1, 300), rng.uniform(-np.pi, np.pi, 300), rng.uniform(-np.pi, np.pi, 300)
    rotated = elements_in_frame(inc, node, omega, FRAME_INVARIABLE)[0]
    np.testing.assert_allclose(rotated, inclination_to_invariable_plane(inc, node), atol=1e-12)


def test_the_ecliptic_frame_returns_the_elements_and_an_unknown_frame_is_refused():
    inc, node, omega = np.array([0.4]), np.array([1.0]), np.array([2.0])
    for got, want in zip(elements_in_frame(inc, node, omega, FRAME_ECLIPTIC), (inc, node, omega)):
        np.testing.assert_array_equal(got, want)
    with pytest.raises(ValueError):
        elements_in_frame(inc, node, omega, 'galactic')


def test_at_30_deg_the_frames_give_the_same_turns():
    # a ZLK-like orbit at i ~ 30 deg: the frames shift omega by a few degrees, never by a turn
    t = np.linspace(0.0, 200_000.0, 20_001)
    inc = np.radians(30.0 + 2.0 * np.sin(2 * np.pi * t / 30_000.0))
    node = -2 * np.pi * t / 60_000.0
    omega = 2 * np.pi * t / 50_000.0 + 0.4 * np.sin(2 * np.pi * t / 9_000.0)
    assert turns(argument_of_pericentre(inc, node, omega)) == pytest.approx(turns(omega), abs=0.05)
    assert wrapped_difference(argument_of_pericentre(inc, node, omega), omega).max() < np.radians(5.0)


def test_at_150_deg_omega_has_no_jump():
    # a retrograde orbit: built from omega (never REBOUND's pomega or l), the rotated omega follows the ecliptic one
    t = np.linspace(0.0, 200_000.0, 20_001)
    inc = np.radians(150.0 + 3.0 * np.sin(2 * np.pi * t / 40_000.0))
    node = 1.0 + 2 * np.pi * t / 70_000.0
    omega = -2 * np.pi * t / 45_000.0
    rotated = np.unwrap(argument_of_pericentre(inc, node, omega))
    assert np.abs(np.diff(rotated)).max() < 3 * np.abs(np.diff(omega)).max()
    assert turns(rotated) == pytest.approx(turns(omega), abs=0.05)
    np.testing.assert_allclose(np.degrees(elements_in_invariable_plane(inc, node, omega)[0]), 150.0, atol=3.0 + 1.6)


# ------------------------------------------------------------------------------------------- the analysis path


def test_a_low_tilt_node_circulates_from_the_invariable_plane_and_is_bounded_from_the_ecliptic():
    t, inc, node, omega = low_tilt_orbit()
    assert np.ptp(np.unwrap(omega)) < 2 * np.pi  # the ecliptic omega stays within a turn
    body, resonance = zlk_body(t, np.full_like(t, 0.03), inc, node, omega)

    ecliptic = long_period_series(body, resonance, FRAME_ECLIPTIC)
    invariable = long_period_series(body, resonance, FRAME_INVARIABLE)
    assert ecliptic.frame == FRAME_ECLIPTIC and invariable.frame == FRAME_INVARIABLE
    np.testing.assert_allclose(ecliptic.angle, body.angles_unwrapped[resonance.to_s()])
    assert turns(invariable.angle) == pytest.approx(8.0, abs=0.05)  # 200 kyr / 25 kyr: omega = varpi - Omega, Omega falls
    assert invariable.frame_shift_turns >= 1.0

    bounded = analyse_zlk(ecliptic, SETTINGS, gate_config=NO_GATE)
    circulating = analyse_zlk(invariable, SETTINGS, gate_config=NO_GATE)
    assert bounded.verdict.angle_status != AngleStatus.CIRCULATION
    assert Rule.ZLK_INVARIABLE_FRAME not in bounded.verdict.rules_fired
    assert circulating.verdict.angle_status == AngleStatus.CIRCULATION
    assert circulating.verdict.status == Status.NON_RESONANT
    assert Rule.ZLK_INVARIABLE_FRAME in circulating.verdict.rules_fired
    row = circulating.numbers_row()
    assert row['zlk_angle_frame'] == FRAME_INVARIABLE and row['zlk_frame_shift_turns'] >= 1.0


def test_the_series_holds_the_invariable_elements_and_the_angle_is_written_back():
    t, inc, node, omega = low_tilt_orbit(n=2_001)
    body, resonance = zlk_body(t, np.full_like(t, 0.03), inc, node, omega)
    series = long_period_series(body, resonance, FRAME_INVARIABLE)
    np.testing.assert_allclose(series.inc, inclination_to_invariable_plane(inc, node), atol=1e-12)
    np.testing.assert_allclose(series.inc, np.radians(1.2), atol=1e-9)
    assert wrapped_difference(series.varpi - series.Omega, series.angle).max() < 1e-9
    _store_zlk_angle(body, resonance.to_s(), series)
    np.testing.assert_allclose(body.angles_unwrapped[resonance.to_s()], series.angle)
    assert wrapped_difference(body.angles[resonance.to_s()], argument_of_pericentre(inc, node, omega)).max() < 1e-12


def test_a_backward_run_keeps_the_body_row_order():
    t, inc, node, omega = low_tilt_orbit(n=2_001)
    body, resonance = zlk_body(-t, np.full_like(t, 0.03), inc, node, omega)  # decreasing times
    series = long_period_series(body, resonance, FRAME_INVARIABLE)
    assert np.all(np.diff(series.t_yr) > 0)
    _store_zlk_angle(body, resonance.to_s(), series)
    assert wrapped_difference(body.angles[resonance.to_s()], argument_of_pericentre(inc, node, omega)).max() < 1e-12


def test_a_body_without_omega_keeps_the_stored_ecliptic_angle():
    t, inc, node, omega = low_tilt_orbit(n=2_001)
    body, resonance = zlk_body(t, np.full_like(t, 0.03), inc, node, omega)
    body.omega = None
    series = long_period_series(body, resonance, FRAME_INVARIABLE)
    assert series.frame == FRAME_ECLIPTIC
    np.testing.assert_allclose(series.angle, body.angles_unwrapped[resonance.to_s()])


def test_the_ei_coupling_is_not_rotated_twice():
    # sqrt(1 - e^2) cos i' = H in the invariable plane (the ZLK law), node turning: the ecliptic route rotates once,
    # the invariable series is read as it is, and both give the same numbers; rotating it again would not
    t = np.arange(0.0, 200_001.0, 50.0)
    e = 0.3 + 0.05 * np.cos(2 * np.pi * t / 20_000.0)
    h = np.sqrt(1 - 0.3**2) * np.cos(np.radians(50.0))
    inc_inv = np.arccos(h / np.sqrt(1 - e**2))
    node_inv = -2 * np.pi * t / 90_000.0
    inc, node, _ = elements_from_invariable_plane(inc_inv, node_inv, np.zeros_like(t))
    once = diagnostics.measure_ei_coupling(t, e, inc, node, 20_000.0, 'libration', SETTINGS, frame=FRAME_ECLIPTIC)
    as_is = diagnostics.measure_ei_coupling(t, e, inc_inv, node_inv, 20_000.0, 'libration', SETTINGS, frame=FRAME_INVARIABLE)
    assert once.coupling == as_is.coupling == 'zlk'
    for name in ('phase_deg', 'band_r', 'amp_e', 'amp_i_deg', 'ratio_h', 'ratio_beat'):
        assert getattr(as_is, name) == pytest.approx(getattr(once, name), rel=1e-9, abs=1e-12), name
    twice = diagnostics.measure_ei_coupling(t, e, inc_inv, node_inv, 20_000.0, 'libration', SETTINGS, frame=FRAME_ECLIPTIC)
    assert twice.amp_i_deg != pytest.approx(once.amp_i_deg, rel=1e-3)


# ------------------------------------------------------------------------------- benchmark rows (local data only)

TOOLS = Path(__file__).resolve().parents[3] / 'benchmark' / 'tools'

# the e-i numbers of the run before W28 (benchmark/data/test-benchmark-secular-zlk/summary.csv), ecliptic frame
EI_BEFORE = {
    'zlk__3040__LK': (40350.0, 179.672404, -0.996177, 0.124509, 3.010938),
    'zlk__1981__LK': (26190.0, 180.018310, -0.998158, 0.111522, 4.424674),
}


def _classified(example_id, frame):
    sys.path.insert(0, str(TOOLS))
    import bench_io as bio
    import reclassify_secular_zlk as rz

    if not (bio.DATA / 'examples' / example_id).is_dir():
        pytest.skip(f'no saved example {example_id} in {bio.DATA}')
    example = bio.load_example(example_id, with_planets=False)
    sim, body, resonance = rz.build_simulation(example)
    sim.config.zlk_angle_frame = frame
    sim.identify_librations()
    return body, resonance, body.secular_analyses[resonance.to_s()], sim.config.analysis_settings


@pytest.mark.benchmark
@pytest.mark.parametrize('example_id', ['zlk__555527__LK', 'zlk__555527__LK__500kyr'])
def test_555527_circulates_from_the_invariable_plane(example_id):
    # g09-1 A: both records non_resonant/circulation/certain, certain resting on the direct circulation of omega
    body, resonance, analysis, _ = _classified(example_id, FRAME_INVARIABLE)
    verdict = body.librations[resonance.to_s()]
    assert (str(verdict.status), str(verdict.angle_status), str(verdict.reason)) == ('non_resonant', 'circulation', 'circulation')
    assert str(verdict.confidence) == 'certain'
    assert Rule.ZLK_INVARIABLE_FRAME in verdict.rules_fired and analysis.series.frame_shift_turns >= 1.0


@pytest.mark.benchmark
@pytest.mark.parametrize('example_id', sorted(EI_BEFORE))
def test_the_ei_numbers_of_3040_and_1981_are_unchanged(example_id):
    period, phase, band_r, amp_e, amp_i = EI_BEFORE[example_id]
    _, _, ecliptic, _ = _classified(example_id, FRAME_ECLIPTIC)
    ei = ecliptic.zlk.ei
    assert (ei.period_yr, ei.coupling) == (pytest.approx(period), 'zlk')
    assert (ei.phase_deg, ei.band_r, ei.amp_e, ei.amp_i_deg) == pytest.approx((phase, band_r, amp_e, amp_i), abs=1e-6)
    # the invariable series at the same period gives the same numbers (its inclination is not rotated again); the
    # period as the ecliptic run measured it, bit for bit (one sample more or less at the edge moves the fit by 1e-4)
    _, _, invariable, settings = _classified(example_id, FRAME_INVARIABLE)  # the record's own settings
    s = invariable.series
    again = diagnostics.measure_ei_coupling(s.t_yr, s.e, s.inc, s.Omega, ei.period_yr, 'libration', settings, frame=s.frame)
    assert (again.phase_deg, again.band_r, again.amp_e, again.amp_i_deg) == pytest.approx((phase, band_r, amp_e, amp_i), abs=1e-6)
