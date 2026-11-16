"""W15: the alias gate judges the lambda part of the angle; the body's varpi takes the winding of its slow vector.

Reference numbers are worked out by hand in the comments:
- an e-vector moving on a straight line at v = 1e-3 per yr that passes the origin at d = 1e-4 turns varpi by
  2 atan(v dt / 2d) = 2 atan(5) = 157.4 deg in the 1-yr step across the passage (> MMR_UNWRAP_MAX_RESID_DEG = 135),
  while lambda, k n dt = 0.17 turn per step, stays resolved: the old gate called that an alias, the new one may not;
- a 10-yr loop of radius r = 0.003 around a slow vector of length 0.002 (r > |z_f|) encircles the origin once per
  loop: np.unwrap adds 100 turns in 1000 yr, the slow vector (2 pi / 2000 yr) half a turn; its phase scatter is
  r / |z_f| = 1.5 > MMR_JITTER_MAX_RAD (pi/4): undefined everywhere.
"""

import math

import numpy as np
import pytest

from resonances.data import const
from resonances.resonance.analysis.mmr import context as ctx
from resonances.resonance.analysis.mmr import kinematics
from resonances.resonance.analysis.mmr import sampling as smp
from resonances.resonance.analysis.mmr.models import MMRSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.settings import settings_from_config

DEG = math.pi / 180
SPEC = ResonanceSpec.from_key('2J-1+0-1')  # sigma = 2 lambda_J - lambda - varpi, p = -1
N_JUPITER = const.PLANET_N_DEG_PER_GAUSS_YR['Jupiter'] * DEG
A_BODY = 3.15  # au: inside the 2:1 at 3.28 au, sigma circulates at ~ -17 deg/yr, 1-yr step resolved


def _series(z: np.ndarray, t: np.ndarray, a: np.ndarray, spec: ResonanceSpec = SPEC, a_body: float = A_BODY) -> MMRSeries:
    """A circulating record of kJ+k+0+p (default 2J-1+0-1) of a body with e-vector z (no planets saved: the angle needs
    only the body's varpi)."""
    n = float(ctx.mean_motion(a_body))
    body_lambda = n * t + 0.3
    varpi = np.angle(z)
    k_j, k, p = spec.planet_coefficients[0], spec.body_coefficient, spec.body_varpi_coefficient
    sigma = k_j * (N_JUPITER * t + 1.1) + k * body_lambda + p * varpi
    ones = np.ones_like(t)
    Omega = 0.7 * ones
    return MMRSeries(
        t, a, np.abs(z), 0.05 * ones, Omega, np.mod(varpi - Omega, 2 * math.pi), np.mod(varpi, 2 * math.pi),
        np.mod(sigma, 2 * math.pi), spec, Regime(float(t[1] - t[0]), float(t[-1] - t[0]), len(t)), {},
        np.mod(body_lambda, 2 * math.pi), name='w15-synthetic',
    )  # fmt: skip


def _passage_series(a_noise_au: float = 0.0) -> MMRSeries:
    """z on a straight line, 1e-3 per yr, passing the origin at 1e-4 at t = 2000.5 yr (mid-step); 4000 yr at a 1-yr step."""
    t = np.arange(0.0, 4000.0 + 0.5, 1.0)
    z = 1e-3 * (t - 2000.5) + 1j * 1e-4
    rng = np.random.default_rng(15)
    a = A_BODY + 1e-5 * np.sin(2 * math.pi * t / 7.3) + a_noise_au * rng.standard_normal(len(t))
    return _series(z, t, a)


def _sampling(series, **overrides):
    settings = settings_from_config(**overrides)
    return smp.analyse_sampling(series, ctx.build_context(series, settings), settings), settings


def test_a_fast_varpi_turn_at_e_near_zero_is_no_alias():
    series = _passage_series()
    sampling, settings = _sampling(series)
    # the full angle turns by 2 atan(5) = 157.4 deg in the step across the passage; the lambda part not at all
    assert math.degrees(sampling.unwrap_resid_full_max_rad) == pytest.approx(157.4, abs=1.0)
    assert math.degrees(sampling.unwrap_resid_max_rad) < 1.0
    assert sampling.unwrap_ok and not sampling.undefined_phase
    assert sampling.n_slow_element_steps >= 1 and sampling.n_lambda_suspect_steps == 0
    assert 'varpi_fast_low_e' in sampling.flags and 'unwrap_step_suspect' not in sampling.flags
    assert not sampling.alias_numpy_differs
    # the stitched angle is still the saved angle mod 2pi at every sample
    assert np.max(np.abs(np.angle(np.exp(1j * (sampling.angle_unwrapped - series.angle_wrapped))))) < 1e-9
    columns = sampling.to_columns()
    assert columns['smp_unwrap_resid_full_max_deg'] == pytest.approx(157.4, abs=1.0)
    assert columns['smp_n_slow_element_steps'] == sampling.n_slow_element_steps


def test_a_true_alias_of_the_lambda_part_stays_an_alias():
    # the same record with sigma scrambled by up to +-170 deg per sample: the lambda part cannot be stitched
    series = _passage_series()
    rng = np.random.default_rng(3)
    scrambled = np.mod(series.angle_wrapped + rng.uniform(-170, 170, len(series.t_yr)) * DEG, 2 * math.pi)
    sampling, _ = _sampling(MMRSeries(**{**series.__dict__, 'angle_wrapped': scrambled}))
    assert not sampling.unwrap_ok and not sampling.undefined_phase
    assert math.degrees(sampling.unwrap_resid_p99_rad) > 135
    assert sampling.n_lambda_suspect_steps > 0 and 'unwrap_step_suspect' in sampling.flags


def _loop_series():
    """A slow vector of length 0.002 turning once in 2000 yr, plus a 10-yr loop of radius 0.003 around it; 1000 yr."""
    t = np.arange(0.0, 1000.0 + 0.05, 0.1)
    z = 0.002 * np.exp(1j * 2 * math.pi * t / 2000.0) + 0.003 * np.exp(1j * 2 * math.pi * t / 10.0)
    return t, z


def test_slow_vector_unwrap_drops_the_turns_of_short_period_loops():
    t, z = _loop_series()
    varpi = np.angle(z)
    wound, scatter = smp.slow_vector_unwrap(t, np.abs(z), varpi, 50.0, 4)
    assert (np.unwrap(varpi)[-1] - np.unwrap(varpi)[0]) / (2 * math.pi) == pytest.approx(100.5, abs=0.6)
    # the slow vector: half a turn in 1000 yr (the deviation from it stays within +-pi)
    assert (wound[-1] - wound[0]) / (2 * math.pi) == pytest.approx(0.5, abs=0.6)
    assert np.max(np.abs(np.angle(np.exp(1j * (wound - varpi))))) < 1e-9  # exact mod 2pi (round-off of 100 turns)
    # r / |z_f| = 0.003 / 0.002 away from the record's edges
    inner = (t > 100) & (t < 900)
    assert np.median(scatter[inner]) == pytest.approx(1.5, rel=0.05)


def test_slow_vector_unwrap_is_np_unwrap_for_a_well_defined_vector():
    t = np.arange(0.0, 5000.0 + 0.5, 1.0)
    z = 0.1 * np.exp(1j * 2 * math.pi * t / 3000.0) + 0.004 * np.exp(1j * 2 * math.pi * t / 11.86)
    varpi = np.angle(z)
    wound, scatter = smp.slow_vector_unwrap(t, np.abs(z), varpi, 50.0, 4)
    assert np.array_equal(wound, np.unwrap(varpi))  # bit for bit: the numbers of every other record stay
    assert np.nanmax(scatter[100:-100]) < 0.06  # ~ 0.004 / 0.1 = 0.04


def test_slow_vector_unwrap_needs_a_band_longer_than_two_steps_and_slower_turning():
    t, z = _loop_series()
    varpi = np.angle(z)
    for band in (0.15, 1e6):  # band <= 2 steps; the vector turns by >= pi per band
        wound, scatter = smp.slow_vector_unwrap(t, np.abs(z), varpi, band, 4)
        assert np.array_equal(wound, np.unwrap(varpi)) and np.all(np.isnan(scatter))


def test_loops_around_the_origin_make_the_phase_undefined_locally_and_lambda_part_consistent():
    t, z = _loop_series()
    series = _series(z, t, A_BODY + 1e-5 * np.sin(2 * math.pi * t / 7.3))
    sampling, settings = _sampling(series)
    assert sampling.unwrap_ok and not sampling.undefined_phase
    assert sampling.phase_undefined_fraction > 0.9 and 'phase_undefined_local' in sampling.flags
    assert sampling.phase_undefined_longest_yr > 800
    # psi = sigma - p varpi with the same varpi winding: the lambda part has no 2pi steps (np.unwrap's varpi has 100)
    psi = kinematics.lambda_part(series, sampling.angle_unwrapped, settings)
    n = float(ctx.mean_motion(A_BODY))
    assert np.max(np.abs(np.diff(psi) - (2 * N_JUPITER - n) * 0.1)) < 1e-3
    old = kinematics.lambda_part(series, sampling.angle_unwrapped)
    assert np.max(np.abs(np.diff(old))) > math.pi


def test_undefined_phase_of_the_whole_record_stays_with_jitter_low_e():
    # 3J-1+0-2 (p = -2) with an e-vector of random direction and Rayleigh length (std/mean = 0.52, x |p| = 1.05 >
    # pi/4: jitter_low_e) and a varpi that jumps at random: the full angle fails and the record is undefined_phase, as
    # before W15; the lambda part alone would pass
    t = np.arange(0.0, 2000.0 + 0.5, 1.0)
    rng = np.random.default_rng(5)
    z = 0.002 * (rng.standard_normal(len(t)) + 1j * rng.standard_normal(len(t)))
    series = _series(z, t, 2.45 + 1e-5 * np.sin(2 * math.pi * t / 7.3), ResonanceSpec.from_key('3J-1+0-2'), 2.45)
    sampling, _ = _sampling(series)
    assert sampling.jitter_low_e and math.degrees(sampling.unwrap_resid_full_max_rad) > 135
    assert math.degrees(sampling.unwrap_resid_max_rad) < 1.0
    # W20: varpi itself is undefined (its own scatter ~ 1 > pi/4 on most samples), not only p varpi
    assert sampling.varpi_undefined_fraction > 0.5
    assert sampling.undefined_phase and not sampling.unwrap_ok


def test_scatter_is_not_measured_at_the_edges_and_infinite_for_a_zero_vector():
    # a constant vector turning once per 100 yr (Codex W15: its filtered edge gave s/|z_f| = 0.95 > pi/4 for 10.6 yr)
    t = np.arange(0.0, 1000.0 + 0.05, 0.1)
    angle = 2 * math.pi * t / 100.0
    _, scatter = smp.slow_vector_unwrap(t, 0.05 * np.ones_like(t), angle, 50.0, 4)
    edge = (t < 50.0) | (t > 950.0)
    assert np.all(np.isnan(scatter[edge]))
    # gain of the 100-yr turn through the 50-yr filter: interior median 0.0039 (Codex W15), nowhere near pi/4
    assert np.median(scatter[~edge]) == pytest.approx(0.0039, abs=0.0005) and np.max(scatter[~edge]) < 0.1
    # e = 0 throughout: varpi undefined everywhere, not "not measured"
    _, zero = smp.slow_vector_unwrap(t, np.zeros_like(t), np.zeros_like(t), 50.0, 4)
    assert np.all(np.isinf(zero[~edge]))
