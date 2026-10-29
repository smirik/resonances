"""W20 CS6 (F5): the force law above a binding line of a_dot with its null test, the condition of the cap
short_period_aliased, and the energy of the turning points in the averaged well.

Independent numbers:
- the synthetic pendulum is the exact solution sin(sigma/2) = k sn(omega0 t, k) (Jacobi), period 4 K(k) / omega0, with
  a = a* + sigma_dot / c, so a_dot = -(omega0^2 / c) sin sigma exactly (K = omega0^2 / |c|); the line at 64 yr is added
  to a with an a_dot amplitude L. The expected R^2 of the force law is var(K sin sigma) / (var(K sin sigma) + (L G)^2 / 2)
  with G the gain of the filters at 64 yr, G(P; P_c) = 1 / (1 + (tan(pi dt / P) / tan(pi dt / P_c))^4) for the
  second-order filtfilt (FILTER_ORDER = 2): 0.564 at the base cutoff 60 yr (0.1 P_e), 0.165 at 1.5 x 64 = 96 yr;
- the energy in a well: <D> = -cos sigma has its bottom at 0 and its barrier at pi, h = (1 - cos sigma) / 2 (0.25 at
  60 deg, 0.5 at 90 deg); <D> = -cos sigma - 0.5 cos 2 sigma has its minimum at 0 (-1.5) and its maxima at
  cos sigma = -1/2 (sigma = 120 deg, <D> = 0.75): h(60 deg) = (-0.5 + 0.25 + 1.5) / 2.25 = 0.5556;
- benchmark: 2004 TK14 31M-34 (B 2026-10-19, its own averaging script): h in the averaged well median 0.07, max 0.59,
  at 36 TPs; the 0.5-yr record's force law in the windows of the 11.1-19.0 kyr episode R^2 0.72-0.94 at a 150-yr
  cutoff, null (shifted a) at most 0.27; Codex: a line near 63.9 yr. 433 Eros 47M-44: lines near 41 and 82-84 yr
  (Codex 19.10), R^2 0.60-0.70 / 0.82-0.90 at 150-200 yr (Claude), null at most 0.35.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import ellipj, ellipk

from resonances.resonance.analysis import filtering
from resonances.resonance.analysis.confidence import Cap
from resonances.resonance.analysis.mmr import averaged_potential, io, response, zone
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.pendulum import CycleClass, CycleEnergy, Zone, ZoneMethod
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import CheckValue

S = settings_from_config()
PERIOD = 600.0
LINE = 64.0
STEP = 1.0
SPAN = 20000.0
P_C = 20.0
AMPLITUDE = math.radians(68.9)  # J0(2A) = 0: var(sin sigma) = 1/2 for a harmonic swing
C = 1.0  # sigma_dot = c (a - a*)


def gain(period, cutoff, step=STEP):
    return 1 / (1 + (math.tan(math.pi * step / period) / math.tan(math.pi * step / cutoff)) ** 4)


def pendulum(line_ratio: float):
    """(t, sigma, a, K): the exact pendulum of period PERIOD and half-amplitude AMPLITUDE, a = sigma_dot / c, plus a line
    of a_dot amplitude line_ratio x K at LINE yr."""
    t = np.arange(0.0, SPAN + STEP / 2, STEP)
    k = math.sin(AMPLITUDE / 2)
    omega0 = 4 * ellipk(k**2) / PERIOD
    sn, cn, _, _ = ellipj(omega0 * t, k**2)
    sigma = 2 * np.arcsin(k * sn)
    rate = 2 * k * omega0 * cn
    force = omega0**2 / abs(C)
    a = 1.0 + rate / C + line_ratio * force * LINE / (2 * math.pi) * np.sin(2 * math.pi * t / LINE + 0.3)
    return t, sigma, a, force


def prepared(t, sigma, a):
    """The series as `preprocess` would hand them over: low-passed at P_c, derivatives of the filtered series."""
    angle_f = filtering.lowpass(t, sigma, P_C, S.filter_order)
    a_f = filtering.lowpass(t, a, P_C, S.filter_order)
    return SimpleNamespace(
        t_yr=t,
        angle_f=angle_f,
        rate_f=filtering.time_derivative(t, angle_f),
        a_f=a_f,
        adot_f=filtering.time_derivative(t, a_f),
        angle_clean=sigma,
        cutoff=SimpleNamespace(cutoff_yr=P_C),
    )


def measure(prep, settings=S):
    pieces = [(prep.t_yr[0] + 2 * PERIOD, prep.t_yr[-1] - 2 * PERIOD)]
    judge = response.ResponseJudge(aliased=False, too_slow=False, sampling=SimpleNamespace(), series=None)
    context = SimpleNamespace(identity_coefficient=C)
    return response.measure_pieces(prep, context, settings, pieces, PERIOD, (AMPLITUDE, AMPLITUDE), judge=judge)


def expected_r2(sigma, force, line_ratio, cutoff):
    signal = np.var(force * np.sin(sigma))
    leak = (line_ratio * force * gain(LINE, P_C) * gain(LINE, cutoff)) ** 2 / 2
    return signal / (signal + leak)


# --------------------------------------------------------------------------------------------- 6a, the line rule


def test_a_line_of_a_dot_raises_the_cutoff_and_the_law_holds_above_it():
    t, sigma, a, force = pendulum(1.5)
    prep = prepared(t, sigma, a)
    base = response.response_cutoff_yr(P_C, PERIOD, S)
    assert base == pytest.approx(60.0)
    # the base series: R^2 near the hand value (0.54) and below 0.6, and the 64-yr line found at 0.84 K
    angle_b, _, _, adot_b, _ = response.smoothed_series(prep, base, S)
    windows = response._force_law(t, angle_b, adot_b, [(1200.0, SPAN - 1200.0)], 3.5 * PERIOD, S)
    r2_base = min(response._halves(windows))
    assert r2_base == pytest.approx(expected_r2(sigma, force, 1.5, base), abs=0.05) and r2_base < 0.6
    cutoff, line = response.line_margin_cutoff(t, adot_b, [(1200.0, SPAN - 1200.0)], PERIOD, base, force, S)
    assert line == pytest.approx(LINE, rel=0.01) and cutoff == pytest.approx(1.5 * LINE, rel=0.01) and cutoff <= PERIOD / 4

    measured = measure(prep)
    assert measured.force_cutoff_yr == pytest.approx(1.5 * LINE, rel=0.01)
    assert measured.force_line_binding_yr == pytest.approx(LINE, rel=0.01)
    assert min(measured.force_r2_h1, measured.force_r2_h2) == pytest.approx(expected_r2(sigma, force, 1.5, 96.0), abs=0.04)
    assert min(measured.force_r2_h1, measured.force_r2_h2) >= 0.9
    assert measured.force_r2_null_max < 0.3
    assert measured.force_null_gain >= S.mmr_force_null_gain
    assert measured.a_response == CheckValue.PASS and not measured.a_response_thin
    assert measured.force_amplitude_au_yr == pytest.approx(force, rel=0.05)


def test_a_clean_pendulum_keeps_its_cutoff_and_its_numbers():
    t, sigma, a, force = pendulum(0.0)
    prep = prepared(t, sigma, a)
    measured = measure(prep)
    assert measured.force_cutoff_yr == pytest.approx(60.0) and math.isnan(measured.force_line_binding_yr)
    assert math.isnan(measured.force_r2_null_max) and math.isnan(measured.force_null_gain)
    assert measured.a_response == CheckValue.PASS and min(measured.force_r2_h1, measured.force_r2_h2) > 0.99
    # the rule switched off (no line can reach 1e9 K): every number of the measurement is the same
    off = measure(prep, settings_from_config(mmr_response_line_amp_frac=1e9))
    for name in ('force_r2_h1', 'force_r2_h2', 'force_amplitude_au_yr', 'phi0_rad', 'identity_slope_ratio', 'b1_ratio'):
        assert getattr(measured, name) == getattr(off, name)
    assert measured.a_response == off.a_response and len(measured.windows) == len(off.windows)


def test_a_law_no_better_than_a_shifted_a_fails():
    # a demanding null gain (1.0: R^2 would have to exceed the null by the whole scale) turns the raised pass into a fail
    t, sigma, a, _ = pendulum(1.5)
    measured = measure(prepared(t, sigma, a), settings_from_config(mmr_force_null_gain=1.0))
    assert np.isfinite(measured.force_line_binding_yr)
    assert measured.force_null_gain < 1.0 and measured.a_response == CheckValue.FAIL


def test_adot_lines_leave_out_the_libration_harmonics():
    # a_dot = sin(2 pi t / 600) + 0.8 sin(2 pi t / 200) (3rd harmonic) + 0.6 sin(2 pi t / 130): only 130 yr is a line
    t = np.arange(0.0, 6000.0, 1.0)
    adot = np.sin(2 * np.pi * t / 600) + 0.8 * np.sin(2 * np.pi * t / 200) + 0.6 * np.sin(2 * np.pi * t / 130)
    lines = response.adot_lines(t, adot, [(0.0, 5999.0)], 600.0, 30.0)
    strong = [(period, amplitude) for period, amplitude in lines if amplitude > 0.3]
    assert len(strong) == 1
    assert strong[0][0] == pytest.approx(130.0, rel=0.01) and strong[0][1] == pytest.approx(0.6, rel=0.05)


# --------------------------------------------------------------------------------------------- 6b, the cap
# The cap short_period_aliased after W20 round D: tests/resonances/analysis/test_w20_round_d.py


# --------------------------------------------------------------------------------------------- 6c, the energy in the well


def test_energy_in_a_cosine_well():
    sigma = 2 * np.pi * np.arange(72) / 72
    values = -np.cos(sigma)
    assert averaged_potential.energy_in_well(sigma, values, math.radians(60), 0.0) == pytest.approx(0.25, abs=1e-3)
    assert averaged_potential.energy_in_well(sigma, values, math.radians(-90), 0.0) == pytest.approx(0.5, abs=1e-3)
    two = -np.cos(sigma) - 0.5 * np.cos(2 * sigma)
    assert averaged_potential.energy_in_well(sigma, two, math.radians(60), math.radians(10)) == pytest.approx(1.25 / 2.25, abs=2e-3)


def _zone(j_lo):
    return Zone(ZoneMethod.NUMERIC, j_lo, float('nan'), 1.0, 1.0, 1.0, False)


def _two_cycles():
    return [CycleEnergy(0.0, 2.0, False, 0.1, 0.1, 2.0, 1.0), CycleEnergy(2.0, 4.0, False, 0.1, 0.1, 2.0, 1.0)]


def test_averaged_classes_read_the_zone_edge():
    """W25: the turning points' J/J_s in the averaged well (`TurningPointEnergies.j`) against J_lo; h is not read."""
    cycles, kinds = _two_cycles(), [zone.CYCLE_EPISODE, zone.CYCLE_EPISODE]
    times, h = np.array([0.0, 1.0, 2.0, 3.0, 4.0]), np.full(5, 0.9)
    energies = zone.TurningPointEnergies(times, h, np.array([0.05, 0.06, 0.07, 0.3, 0.2]))
    classes, edges, source = zone.averaged_classes(cycles, kinds, [CycleClass.ZONE] * 2, [True, True], energies, _zone(0.25))
    assert classes == [CycleClass.DEEP, CycleClass.ZONE] and edges == [False, False] and source == zone.SOURCE_AVERAGED_POTENTIAL
    # no zone edge: the pendulum's classes stay
    kept = zone.averaged_classes(cycles, kinds, [CycleClass.ZONE] * 2, [True, True], energies, _zone(float('nan')))
    assert kept == ([CycleClass.ZONE] * 2, [True, True], zone.SOURCE_PENDULUM)


def test_a_cycle_with_an_unmeasured_turning_point_keeps_the_pendulum_class():
    """FIX1 (Codex A #5): a cycle's class is DEEP only when every turning point's J/J_s is measured and below J_lo; one
    at or above J_lo makes it ZONE whatever the others are. W25: a loop around several wells of the profile is such an
    unmeasured point."""
    cycles, kinds = _two_cycles(), [zone.CYCLE_EPISODE, zone.CYCLE_EPISODE]
    nan = float('nan')
    times, h = np.array([0.0, 1.0, 2.0, 3.0, 4.0]), np.full(5, 0.5)
    energies = zone.TurningPointEnergies(times, h, np.array([0.05, nan, 0.07, nan, 0.3]))
    classes, edges, source = zone.averaged_classes(cycles, kinds, [CycleClass.ZONE] * 2, [True, True], energies, _zone(0.25))
    # first cycle: 0.05, nan, 0.07 -> not decided by the averaged well; second: 0.07, nan, 0.3 -> zone
    assert classes == [CycleClass.ZONE, CycleClass.ZONE] and edges == [True, False] and source == zone.SOURCE_AVERAGED_POTENTIAL
    nothing = zone.TurningPointEnergies(np.array([0.0, 1.0, 2.0]), np.full(3, 0.5), np.array([0.05, nan, 0.07]))
    assert zone.averaged_classes(cycles[:1], kinds[:1], [CycleClass.ZONE], [True], nothing, _zone(0.25)) == (
        [CycleClass.ZONE],
        [True],
        zone.SOURCE_PENDULUM,
    )


def test_turning_point_energies_do_not_bridge_a_failed_profile(monkeypatch):
    """FIX1 (Codex A #5): 9 turning points at t = 0..8, h evaluated at 5 of them (0, 2, 4, 6, 8) as h = 0.01 t, the one
    at t = 4 failed (nan): t = 1 and 7 are interpolated (0.01, 0.07), t = 3 and 5 next to the failure stay nan. W25:
    J/J_s (here 0.02 t) is interpolated the same way, on its own: J alone nan at t = 6 (a loop around two wells) keeps
    t = 5 and 7 nan while h bridges them. W25 (Codex #3): J_s(t) (here 1 + t) is a third column, interpolated the
    same way."""
    from resonances.resonance.analysis.mmr import centre_track

    def fake(well, time, value, middle):
        if time == 4.0:
            return float('nan'), float('nan'), float('nan')
        return 0.01 * time, float('nan') if time == 6.0 else 0.02 * time, 1.0 + time

    monkeypatch.setattr(centre_track, 'turning_point_energy', fake)
    well = SimpleNamespace(available=True)
    times = np.arange(9.0)
    settings = SimpleNamespace(mmr_separatrix_avgpot_tp_max=5)
    h, j, j_s = centre_track.turning_point_energies(well, times, np.zeros(9), settings, np.zeros(9))
    nan = float('nan')
    assert h == pytest.approx([0.0, 0.01, 0.02, nan, nan, nan, 0.06, 0.07, 0.08], nan_ok=True)
    assert j == pytest.approx([0.0, 0.02, 0.04, nan, nan, nan, nan, nan, 0.16], nan_ok=True)
    assert j_s == pytest.approx([1.0, 2.0, 3.0, nan, nan, nan, 7.0, 8.0, 9.0], nan_ok=True)


# --------------------------------------------------------------------------------------------- benchmark


def _analysis(example):
    if not (io.example_path(example) / 'series.csv.gz').exists():
        pytest.skip('benchmark data not available')
    return analyse_mmr(io.load_series(example), S)


@pytest.mark.benchmark
def test_tk14_energy_in_the_averaged_well():
    analysis = _analysis('mmr2__2004_TK14__31M-34+0+3')
    row = analysis.numbers_row()
    assert row['sep_zone_source'] == 'averaged_potential'
    assert row['sep_h_avgpot_med'] == pytest.approx(0.07, abs=0.05) and row['sep_h_avgpot_max'] <= 0.6
    assert row['sep_h_avgpot_n'] >= 30  # the TPs of the episode cycles (B: all 36 TPs)
    assert set(analysis.energy.cycle_classes) == {CycleClass.DEEP}  # far below the eq_only zone (h_lo 0.99)
    # W20 round D: the record no longer lifts the cap (its fast residual << 34 deg did before)
    assert str(Cap.SHORT_PERIOD_ALIASED) in analysis.verdict.caps
    analysis.drop_arrays()  # W20 round C (review B #8): the per-TP h goes, its columns stay frozen
    assert analysis.energy.h_avgpot == () and analysis.energy.j_avgpot == ()
    assert {k: analysis.numbers_row()[k] for k in row if k.startswith(('sep_h_avgpot', 'sep_J_avgpot'))} == {
        k: row[k] for k in row if k.startswith(('sep_h_avgpot', 'sep_J_avgpot'))
    }


@pytest.mark.benchmark
def test_tk14_half_year_force_law_above_the_64_yr_line():
    analysis = _analysis('mmr2__2004_TK14__31M-34+0+3__step0.5yr')
    read = analysis.response
    assert read.a_response == CheckValue.PASS and read.force_line_binding_yr == pytest.approx(63.9, abs=1.0)
    assert min(read.force_r2_h1, read.force_r2_h2) >= 0.7 and read.force_r2_null_max <= 0.35


@pytest.mark.benchmark
def test_433_force_law_above_the_83_yr_line():
    analysis = _analysis('mmr2__433__47M-44+0-3')
    read = analysis.response
    assert read.force_line_binding_yr == pytest.approx(83.0, abs=2.0)
    assert read.force_cutoff_yr == pytest.approx(1.5 * read.force_line_binding_yr)
    # the single law rises from 0.07/0.16 at 75 yr to above 0.5 in both halves, and beats a shifted a
    assert min(read.force_r2_h1, read.force_r2_h2) > 0.5 and read.force_r2_null_max <= 0.45
