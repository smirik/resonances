"""W20 CS3 on benchmark records (skipped without the benchmark data): the potential route against independent numbers
(R1 prototype and Codex 19.10, R5 §7.5), and the decision of the owner rows (R5 §7.6, F7).

Independent numbers: R1 (scratchpad w20-design/R1-right-column, its own averaging script, order-4 filter): 385266 owner
3N-5 K_th 2.62-2.86e-5 au/yr against fitted 2.65-2.81e-5, a-wave 0.0196-0.0199 au, K/K_th 0.96-1.02; 82157 0.98-1.01;
120348 0.99-1.05; 101703 0.98-1.01. Codex 19.10 at fixed snapshots, band [P/2, 2P]: GK3 10E-17 scale 0.9-1.1 (5, 15,
18 kyr), 2001 HW7 5E-16 0.95-1.15 (3, 10, 17 kyr), 78085 2E-7 0.85-1.05 (5, 40, 65 kyr; the 5-kyr window starts at
2.0 kyr, inside the 3.4-kyr band-pass edge the package trims, so it is left out here)."""

import math

import numpy as np
import pytest

from resonances.resonance.analysis import filtering
from resonances.resonance.analysis.mmr import io, owner, owner_potential
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.models import ResonanceSpec
from resonances.resonance.analysis.settings import settings_from_config

S = settings_from_config()


def _analysis(example):
    if not (io.example_path(example) / 'series.csv.gz').exists():
        pytest.skip('benchmark data not available')
    series = io.load_series(example)
    return series, analyse_mmr(series, S)


def _potential_at(series, analysis, key, centres_kyr):
    """The potential route of candidate `key` on windows centred at `centres_kyr` (the frame of the owner search)."""
    prep, context = analysis.prep, analysis.context
    frame = owner.build_frame(prep, series, analysis.owner.freq_period_yr, S, None, context=context)
    planets = owner.available_planets(series, S)
    planet_n = {name: owner.planet_mean_motion(context, name) for name in planets}
    spec = ResonanceSpec.from_key(key)
    t = np.asarray(prep.t_yr, float)
    longitudes = {name: series.planets[name].mean_longitude for name in spec.planets}
    deviations = owner.planet_slow_deviations(t, longitudes, planet_n, prep.cutoff.cutoff_yr, S.filter_order)
    basis = owner.AngleBasis(series, frame, planet_n, deviations, prep.cutoff.cutoff_yr, owner.slope(t, frame.varpi_unwrapped), S)
    starts = [1e3 * c - frame.window_yr / 2 for c in centres_kyr]
    phi_fit = basis.filtered_fit(basis.coefficients(spec))
    return owner_potential.owner_potential(spec, starts, frame, owner.wrapped_angle(spec, series), phi_fit, series, context, S)


def _ratios(fit):
    return np.array([w.k_fit / w.k_theory for w in fit.windows])


# --------------------------------------------------------------------------------------------- the potential route


@pytest.mark.benchmark
def test_385266_owner_force_matches_the_averaged_potential():
    series, analysis = _analysis('mmr2__385266__16N-27+0+11')
    result = analysis.owner
    assert result.accepted.spec.key == '3N-5+0+2' and result.detail == 'owner_forced_wave'
    fit = result.accepted.potential
    assert fit.measured and np.all((_ratios(fit) >= 0.9) & (_ratios(fit) <= 1.1))
    phases = [abs(math.degrees(np.angle(np.exp(1j * (w.phi0_fit - w.phi0_theory))))) for w in fit.windows]
    assert max(phases) <= 3.0
    # R1: K_th 2.62-2.86e-5 au/yr (unfiltered), the forced a-wave 0.0196-0.0199 au
    assert 2.5e-5 <= fit.k_direct <= 2.9e-5
    row = analysis.numbers_row()
    assert row['own_wave_da_pred_au'] == pytest.approx(0.0199, rel=0.1)
    assert row['own_wave_da_meas_au'] == pytest.approx(0.0199, rel=0.1)
    assert row['own_identity_text'] == '(27/5)φ + (ϖ − λ_N)/5'
    # the rank-2 candidate 13N-22+0+9 (R^2 0.83, phase incoherent): its own term is ~1/170 of the force it would need
    negative = _potential_at(series, analysis, '13N-22+0+9', [17.0, 33.0, 49.0, 64.0, 80.0])
    assert negative.measured
    assert not (
        np.all(negative.scales >= 1 / S.mmr_owner_potential_scale_max) and np.all(negative.scales <= S.mmr_owner_potential_scale_max)
    )
    assert owner.potential_accepts(negative, S)[0] is False


@pytest.mark.benchmark
@pytest.mark.parametrize('example', ['mmr2__82157__20N-29+0+9', 'mmr2__120348__15N-22+0+7', 'mmr2__101703__41J-14+0-27'])
def test_two_body_owner_force_matches_the_averaged_potential(example):
    _, analysis = _analysis(example)
    fit = analysis.owner.accepted.potential
    assert fit.measured
    ratios = _ratios(fit)
    assert np.all((ratios >= 0.9) & (ratios <= 1.1)), ratios  # every window (101703: the 3:1 multiplet, R1)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'example, key, centres, low, high',
    [
        ('mmr3__1997_GK3__3M+5S-3+0+0-5', '10E-17+0+7', [5.0, 15.0, 18.0], 0.9, 1.1),
        ('mmr3__2001_HW7__5J+6S-2+0+0-9', '5E-16+0+11', [3.0, 10.0, 17.0], 0.95, 1.15),
        ('mmr3__78085__5J-4S-1+0+0+0', '2E-7+0+5', [40.0, 65.0], 0.85, 1.05),
    ],
)
def test_potential_scale_at_the_codex_snapshots(example, key, centres, low, high):
    series, analysis = _analysis(example)
    fit = _potential_at(series, analysis, key, centres)
    assert fit.measured and len(fit.windows) == len(centres)
    assert fit.r2_median >= 0.9
    assert np.all((fit.scales >= low) & (fit.scales <= high)), fit.scales


# --------------------------------------------------------------------------------------------- the decisions


@pytest.mark.benchmark
def test_2003_tg2_owner_acts_outside_the_episode():
    # 27V-40 librates near 77 kyr; sigma's decisive episode is 29.9-30.6 kyr: not induced (F7), the author leans uncertain
    _, analysis = _analysis('mmr3__2003_TG2__4E-1M-3+0+0+0__100kyr')
    v, result = analysis.verdict, analysis.owner
    assert result.accepted.spec.key == '27V-40+0+13' and result.cover < S.mmr_owner_cover_min
    assert all(b <= 30e3 or a >= 31e3 for a, b, _ in result.intervals)
    assert (str(v.status), str(v.libration_origin)) == ('uncertain', 'unknown') and v.owner == '27V-40+0+13'
    assert 'owner_outside_episode' in v.flags and v.checks['check_mmr_owner'] == 'pass'


@pytest.mark.benchmark
def test_2001_pg14_step_quarter_year_reads_its_own_libration():
    # 27E-47 circulates at 1.47 yr in the episode (343-2813 yr) and librates only after 11.6 kyr (CS3 note 3a)
    _, analysis = _analysis('mmr2__2001_PG14__14E-25+0+11__step0.25yr')
    v, result = analysis.verdict, analysis.owner
    assert result.cover == 0.0 and str(v.libration_origin) != 'induced' and str(v.reason) != 'induced_libration'
    assert 'owner_outside_episode' in v.flags


@pytest.mark.benchmark
def test_2001_hw7_second_capture_is_unresolved():
    _, analysis = _analysis('mmr3__2001_HW7__5J+6S-2+0+0-9')
    v, result = analysis.verdict, analysis.owner
    assert (str(v.status), str(v.reason), v.owner) == ('non_resonant', 'induced_libration', '5E-16+0+11')
    # x = Psi / 8 sweeps ~45 deg (R5: 46): a second capture cannot be excluded
    assert result.capture.verdict == owner.CAPTURE_UNRESOLVED and math.degrees(result.capture.x_sweep_rad) < 90.0
    assert 'second_capture_unresolved' in v.flags and 'second_capture_unresolved' in v.caps
    assert str(v.confidence) in ('medium', 'low')


@pytest.mark.benchmark
def test_1997_gk3_owner_covers_the_episode():
    _, analysis = _analysis('mmr3__1997_GK3__3M+5S-3+0+0-5')
    v, result = analysis.verdict, analysis.owner
    assert (str(v.status), str(v.reason), v.owner) == ('non_resonant', 'induced_libration', '10E-17+0+7')
    assert result.accepted.route in ('phase', 'potential') and result.cover >= S.mmr_owner_cover_min
    # the libration intervals of R5 (0.1-11.3, 12.8-18.5 kyr) with a slip between
    kinds = [kind for _, _, kind in result.intervals]
    assert 'slip' in kinds and kinds.count('libration') >= 2
    starts = [a for a, _, kind in result.intervals if kind == 'libration']
    assert starts[0] < 1e3 and any(12.0e3 < a < 13.5e3 for a in starts)


@pytest.mark.benchmark
def test_2003_ug22_drift_is_induced_by_10e17():
    # the owner's centre moves with varpi, so its single phase drifts; its averaged potential explains a_dot (F3)
    _, analysis = _analysis('mmr3__2003_UG22__8M-4S-7+0+0+3')
    v, result = analysis.verdict, analysis.owner
    assert (str(v.status), str(v.angle_status), str(v.reason), v.owner) == ('non_resonant', 'drift', 'induced_libration', '10E-17+0+7')
    assert result.accepted.route == 'potential' and result.accepted.phi0_coherence < S.mmr_owner_phi0_coh_min
    assert 0.9 <= result.accepted.potential.scale_lo <= result.accepted.potential.scale_hi <= 1.1


def test_lowpass_of_psi_in_the_capture_test_uses_the_band_period():
    # the capture test low-passes Psi at mmr_owner_capture_lowpass_periods x P: a P-periodic wave is gone, a slow one stays
    t = np.arange(0.0, 20000.0, 1.0)
    period = 200.0
    slow = 0.5 * np.sin(2 * np.pi * t / 8000.0)
    psi = slow + 0.3 * np.sin(2 * np.pi * t / period)
    filtered = filtering.lowpass(t, psi, S.mmr_owner_capture_lowpass_periods * period, S.filter_order)
    inner = (t > 2000) & (t < 18000)
    assert np.max(np.abs(filtered - slow)[inner]) < 0.03


@pytest.mark.benchmark
def test_78085_stays_induced_by_2e7():
    # R5 measured x = Psi / 7 sweeping 199 deg over the whole record; the package tests only the owner's block that meets
    # sigma's episode (3.4-81.6 kyr, trimmed by 3 P), where x sweeps less than 180 deg: unresolved, status unchanged
    _, analysis = _analysis('mmr3__78085__5J-4S-1+0+0+0')
    v, result = analysis.verdict, analysis.owner
    assert (str(v.status), str(v.reason), v.owner) == ('non_resonant', 'induced_libration', '2E-7+0+5')
    assert result.cover >= S.mmr_owner_cover_min and result.capture.energy_r2 < S.mmr_owner_r2_min


@pytest.mark.benchmark
@pytest.mark.parametrize('example', ['mmr2__101703__41J-14+0-27', 'mmr2__142694__41J-14+0-27'])
def test_3j1_multiplet_owner_unchanged(example):
    _, analysis = _analysis(example)
    v = analysis.verdict
    assert (str(v.status), str(v.angle_status), str(v.libration_origin), v.owner) == (
        'non_resonant',
        'circulation',
        'induced',
        '3J-1+0-2',
    )


@pytest.mark.benchmark
def test_1627_two_episodes_are_not_each_others_owners():
    # Ivar: sigma 11E-28 and 13E-33 each named the other as owner (both induced). Each owner acts outside the other's
    # decisive episode, so neither takes it (F7) and each row reads its own episode.
    rows = [_analysis(example)[1] for example in ('mmr2__1627__11E-28+0+17', 'mmr2__1627__13E-33+0+20')]
    for analysis in rows:
        assert analysis.owner.cover < S.mmr_owner_cover_min
        assert 'owner_outside_episode' in analysis.verdict.flags
        assert str(analysis.verdict.libration_origin) != 'induced'
