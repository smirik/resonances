"""W3 'perf' package, owner.py: candidates as combinations of a few unwrapped series (W3-M2), the zoom spectrum of the
reference period, the count-based unwrap check, the skip rule and F3 (the mu scale neither accepts nor rejects).

The reference for the basis is the W2 path: each candidate unwrapped by its own model rate (`owner_angle_with_residual`)
and measured at full resolution (`measure_candidate`). The synthetic body is driven by phi_o = 11 lambda_M - 20 lambda
+ 9 varpi librating about pi (A = 0.5 rad, P = 1500 yr); the catalogue angle is 1V+5J-7+0+0+1.
"""

import math
from dataclasses import replace
import tracemalloc

import numpy as np
import pytest

from resonances.resonance.analysis.mmr import owner
from resonances.resonance.analysis.mmr.context import build_context
from resonances.resonance.analysis.mmr.models import MMRSeries, OwnerCandidate, PlanetSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.mmr.preprocess import preprocess
from resonances.resonance.analysis.mmr.sampling import analyse_sampling
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, CheckValue

S = settings_from_config()
WIDE = settings_from_config(mmr_owner_rate_factor=300.0)  # tens of candidates instead of one
TWO_PI = 2 * math.pi
STEP, SPAN, P_LIB, AMPLITUDE = 1.0, 24000.0, 1500.0, 0.5
N_MARS, N_JUPITER, VARPI_RATE = 3.3405, 0.52965, TWO_PI / 50000.0


def _planet(name, t, n, a, phase):
    flat = np.zeros_like(t)
    return PlanetSeries(name, t, a + flat, 0.05 + flat, 0.02 + flat, 1.0 + flat, 2.0 + flat, np.mod(n * t + phase, TWO_PI))


def _series(key='1V+5J-7+0+0+1') -> MMRSeries:
    t = np.arange(0.0, SPAN + STEP / 2, STEP)
    omega = TWO_PI / P_LIB
    phi_o = math.pi + AMPLITUDE * np.sin(omega * t)
    varpi = 0.4 + VARPI_RATE * t
    lambda_mars = N_MARS * t + 0.9
    mean_longitude = (11 * lambda_mars + 9 * varpi - phi_o) / 20
    n = (11 * N_MARS + 9 * VARPI_RATE - AMPLITUDE * omega * np.cos(omega * t)) / 20
    n_venus = 7 * (11 * N_MARS + 9 * VARPI_RATE) / 20 - 5 * N_JUPITER - VARPI_RATE + TWO_PI / 12000.0
    planets = {
        'Venus': _planet('Venus', t, n_venus, 0.723, 0.2),
        'Mars': _planet('Mars', t, N_MARS, 1.524, 0.9),
        'Jupiter': _planet('Jupiter', t, N_JUPITER, 5.203, 2.1),
    }
    spec = ResonanceSpec.from_key(key)
    angle = spec.body_coefficient * mean_longitude + spec.body_varpi_coefficient * varpi
    for name, k in zip(spec.planets, spec.planet_coefficients):
        angle = angle + k * planets[name].mean_longitude
    node = np.mod(2.0 - TWO_PI / 70000.0 * t, TWO_PI)
    series = MMRSeries(
        t_yr=t,
        a=(TWO_PI / n) ** (2 / 3),
        e=0.15 + np.zeros_like(t),
        inc=0.05 + np.zeros_like(t),
        Omega=node,
        omega=np.mod(varpi - node, TWO_PI),
        varpi=np.mod(varpi, TWO_PI),
        angle_wrapped=np.mod(angle, TWO_PI),
        resonance=spec,
        regime=Regime(step_yr=STEP, span_yr=SPAN, n_rows=len(t), planets_saved=tuple(planets)),
        planets=planets,
        mean_longitude=np.mod(mean_longitude, TWO_PI),
        name='synthetic-w3',
    )
    series.validate()
    return series


@pytest.fixture(scope='module')
def record():
    series = _series()
    context = build_context(series, S)
    prep = preprocess(series, context, analyse_sampling(series, context, S), S)
    return series, prep, context


def _frame_and_basis(series, prep, context, settings):
    """The steps of run_owner up to the candidates (the same calls, so the two paths share their inputs)."""
    t = np.asarray(prep.t_yr, float)
    frame = owner.build_frame(prep, series, P_LIB, settings)
    planets = owner.available_planets(series, settings)
    planet_n = {name: owner.planet_mean_motion(context, name) for name in planets}
    a = np.asarray(series.a, float)[frame.mask]
    n_range = (owner.mean_motion(np.max(a)) + frame.body_n_offset, owner.mean_motion(np.min(a)) + frame.body_n_offset)
    specs = owner.enumerate_owners(
        series.resonance, context, P_LIB, planets, settings, period_yr=P_LIB, body_n_range=n_range, body_varpi_rate=frame.body_varpi_rate
    )
    longitudes = {name: series.planets[name].mean_longitude for name in sorted({p for s in specs for p in s.planets})}
    deviations = owner.planet_slow_deviations(t, longitudes, planet_n, prep.cutoff.cutoff_yr, settings.filter_order)
    varpi_rate = owner.slope(t, frame.varpi_unwrapped)
    basis = owner.AngleBasis(series, frame, planet_n, deviations, prep.cutoff.cutoff_yr, varpi_rate, settings)
    return frame, basis, specs, planet_n, deviations, varpi_rate


def test_basis_measures_every_candidate_as_its_own_unwrap(record):
    series, prep, context = record
    frame, basis, specs, planet_n, deviations, varpi_rate = _frame_and_basis(series, prep, context, WIDE)
    assert len(specs) > 30
    for spec in specs:
        phi, residual = owner.owner_angle_with_residual(spec, series, planet_n, frame.body_n_offset, varpi_rate)
        ripple = sum(k * deviations[name] for name, k in zip(spec.planets, spec.planet_coefficients))
        reference = owner.measure_candidate(spec, phi - ripple, frame, series, prep.cutoff.cutoff_yr, WIDE, None, residual)
        coefficients = basis.coefficients(spec)
        fit = owner.fit_on_frame(
            spec,
            basis.filtered_fit(coefficients),
            basis.turns(coefficients),
            basis.rate(coefficients),
            basis.unwrap_ok(coefficients),
            frame,
        )
        assert fit.unwrap_ok == reference.unwrap_ok
        assert fit.rate_rad_yr == pytest.approx(reference.rate_rad_yr, rel=1e-9, abs=1e-12)
        assert [r.regime for r in fit.regimes] == [r.regime for r in reference.regimes]
        for got, expected in zip(fit.regimes, reference.regimes):
            assert got.starts == expected.starts
            assert got.r2_windows == pytest.approx(expected.r2_windows, abs=1e-8)
            # phases of near-empty fits: rounding; nan in a regime whose windows the grid does not resolve (W15)
            assert got.coherence == pytest.approx(expected.coherence, abs=1e-6, nan_ok=True)
            assert got.force_au_yr == pytest.approx(expected.force_au_yr, rel=1e-7)
            assert got.sigma_slow_turns == expected.sigma_slow_turns


def test_basis_angle_is_the_constructed_owner(record):
    series, prep, context = record
    _, basis, _, _, _, _ = _frame_and_basis(series, prep, context, S)
    spec = ResonanceSpec.from_key('11M-20+0+9')
    phi = basis.clean(basis.coefficients(spec))
    expected = math.pi + AMPLITUDE * np.sin(TWO_PI / P_LIB * series.t_yr)
    offset = np.round((phi[0] - expected[0]) / TWO_PI) * TWO_PI
    # the planets' slow deviation of a linear Mars is ~0, so the clean angle is the constructed one
    assert np.max(np.abs(phi - offset - expected)) < 1e-6


POTENTIAL_BUDGET_BYTES = 200e6


def test_run_owner_finds_the_owner_and_keeps_no_candidate_arrays(record):
    series, prep, context = record
    result = owner.run_owner(series, prep, context, P_LIB, S)
    assert result.check == CheckValue.FAIL and result.accepted.spec.key == '11M-20+0+9'
    peaks = {}
    for mode in ('off', 'auto'):
        tracemalloc.start()
        try:
            wide = owner.run_owner(series, prep, context, P_LIB, replace(WIDE, mmr_owner_potential=mode))
            _, peaks[mode] = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert wide.accepted.spec.key == '11M-20+0+9' and wide.n_candidates > 30
    # a fixed number of full-length arrays (~30 at 24 kyr), not one per candidate (> 30 here)
    assert peaks['off'] < 40 * len(series.t_yr) * 8
    # the accepted owner's averaged potential (W20): blocks of LineAverage independent of the record, <= 200 MB (plan)
    assert peaks['auto'] < 40 * len(series.t_yr) * 8 + POTENTIAL_BUDGET_BYTES


# --------------------------------------------------------------------------------------------- helpers


def test_unwrap_check_by_count_equals_the_percentile():
    rng = np.random.default_rng(3)
    for size in (1, 2, 7, 100, 1001, 5000):
        for scale in (5.0, 30.0, 44.0, 60.0):
            residual = np.radians(rng.normal(0.0, scale, size))
            p99 = np.percentile(np.degrees(np.abs(residual)), 99)
            largest = np.max(np.degrees(np.abs(residual)))
            expected = p99 <= S.mmr_unwrap_max_resid_p99_deg and largest <= S.mmr_unwrap_max_resid_deg
            assert owner.unwrap_ok(residual, S) == expected
    # the limit between the two interpolated order statistics: 100 values, p99 = x[98] + 0.01 (x[99] - x[98])
    values = np.radians(np.r_[np.zeros(98), 44.0, 140.0])
    assert not owner.unwrap_ok(values, S)  # the largest error 140 > 135
    values = np.radians(np.r_[np.zeros(98), 44.0, 134.0])
    assert owner.unwrap_ok(values, S)  # p99 = 44 + 0.01 x 90 = 44.9 <= 45
    values = np.radians(np.r_[np.zeros(98), 44.5, 134.0])
    assert not owner.unwrap_ok(values, S)  # 44.5 + 0.01 x 89.5 = 45.395 > 45
    assert not owner.unwrap_ok(np.array([0.1, np.nan]), S)


def test_dominant_period_is_the_padded_spectrum_peak():
    rng = np.random.default_rng(11)
    t = np.arange(0.0, 30001.0, 3.0)  # 10001 samples: the padded length 80008 = 8 x 73 x 137
    values = 2e-4 * np.sin(TWO_PI * t / 1777.0) + 1e-4 * np.sin(TWO_PI * t / 640.0) + 3e-5 * rng.standard_normal(len(t))
    for low, high in ((500.0, 5000.0), (300.0, 900.0), (1000.0, 1100.0)):
        x = values - np.polyval(np.polyfit(t - t[0], values, 1), t - t[0])
        x = x * np.hanning(len(x))
        size = 8 * len(x)
        frequencies = np.fft.rfftfreq(size, d=3.0)
        power = np.abs(np.fft.rfft(x, size)) ** 2
        with np.errstate(divide='ignore'):
            periods = 1.0 / frequencies
        inside = (periods >= low) & (periods <= high)
        expected = periods[np.argmax(np.where(inside, power, -np.inf))]
        assert owner.dominant_period(t, values, low, high) == pytest.approx(expected, rel=1e-12)
    assert owner.dominant_period(t, values, 500.0, 5000.0) == pytest.approx(1777.0, rel=0.01)


def test_search_matters_only_where_the_tree_reads_the_owner():
    assert not owner.owner_search_matters(AngleStatus.CIRCULATION, AngleSubtype.ALIAS)
    assert not owner.owner_search_matters(AngleStatus.UNDEFINED, AngleSubtype.UNPHYSICAL)
    assert not owner.owner_search_matters(AngleStatus.INSUFFICIENT_TIME, AngleSubtype.NONE)
    assert owner.owner_search_matters(AngleStatus.CIRCULATION, AngleSubtype.NONE)  # names a circulation's origin
    assert owner.owner_search_matters(AngleStatus.LIBRATION, AngleSubtype.STATIONARY)
    assert owner.owner_search_matters(AngleStatus.EPISODIC, AngleSubtype.NONE)
    assert owner.owner_search_matters('drift', '')


def _candidate(strength_ratio=0.01, r2=0.9, coherence=0.95):
    return OwnerCandidate(
        spec=ResonanceSpec.from_key('10E-17+0+7'),
        rate_rad_yr=0.0,
        r2_median=r2,
        r2_windows=(r2,) * 4,
        phi0_coherence=coherence,
        psi_share=0.02,
        force_au_yr=1e-6,
        force_theory_au_yr=1e-6 / strength_ratio,
        strength_ratio=strength_ratio,
        rank=0,
        accepted=False,
        reason='',
    )


def test_mu_scale_neither_accepts_nor_rejects():
    """K_o = 1000 K_th: the scale lacks the disturbing-function coefficient, so the owner is judged on R^2 and phase."""
    accepted, reason = owner.accept(_candidate(strength_ratio=1000.0), 0.3, S, sigma_coherence=0.5)
    assert (accepted, reason) == (True, 'accepted:r2')
    # sigma would need 300 x its scale: no route of its own any more, sigma explains a_dot as well as the candidate
    assert owner.better_than_sigma(_candidate(), 0.9, 0.97, 300.0, S) == ''
    assert owner.accept(_candidate(), 0.9, S, sigma_coherence=0.97, sigma_strength_ratio=300.0) == (False, owner.REASON_NOT_BETTER)
    # the scale still orders the candidates (display, owner_probable)
    assert owner.within_scale(_candidate(strength_ratio=5.0), S) and not owner.within_scale(_candidate(strength_ratio=50.0), S)
