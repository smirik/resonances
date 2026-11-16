"""Output step, alias branch and the predictive unwrap (SPEC §3.1, §4.2).

The 1999 UQ numbers come from SAMPLING.md / NUMBERS §2 (prototype on the benchmark records):
nu = 71.97853 deg/yr at a 5.0012222-yr step is one turn per step minus 0.00387 deg/yr, the Nyquist
half-width there is 0.00927 au, the m = 1 ghost sits at 1.0941756 au and the 1-yr rerun counts 3999
revolutions in 20 kyr. Other references are worked out by hand in the comments.
"""

import math

import numpy as np
import pytest

from resonances.data import const
from resonances.resonance.analysis.mmr import context as ctx
from resonances.resonance.analysis.mmr import sampling as smp
from resonances.resonance.analysis.mmr.models import MMRSeries, PlanetSeries, Regime, ResonanceSpec, StepVerdict
from resonances.resonance.analysis.settings import settings_from_config
from tests.resonances.local_data import benchmark_data

DEG = math.pi / 180
TABLE_N = {name: value * DEG for name, value in const.PLANET_N_DEG_PER_GAUSS_YR.items()}
UQ_STEP = 5.0012222
UQ_RATE = 71.97853 * DEG
UQ_A = 1.0941798
UQ_SPEC = ResonanceSpec.from_key('7E+2M-9+0+0+0')
UQ_RATE_RANGE = (70.8425 * DEG, 72.9714 * DEG)  # nu over a 1..99 % of the 1999 UQ record


# --- branches and verdicts ------------------------------------------------------------------------------


def test_apparent_rate_of_1999_uq_at_the_5_yr_step():
    # 71.97853 - 360 / 5.0012222 = -0.0038746 deg/yr (Knezevic & Milani 2001, p. 22; record: -0.00382)
    assert smp.branch_index(UQ_RATE, UQ_STEP) == 1
    assert smp.apparent_rate(UQ_RATE, UQ_STEP) / DEG == pytest.approx(-0.0038746, abs=1e-6)
    # at 1 yr the rate is resolved and shown as is; at 15.915 yr it hides 3 turns
    assert smp.apparent_rate(UQ_RATE, 1.0) == pytest.approx(UQ_RATE)
    assert smp.branch_index(UQ_RATE, 15.915) == 3


def test_branch_index_rounds_half_turns_away_from_the_resolved_side():
    step = 1.0
    turns = np.array([0.49, 0.51, -0.49, -0.51, 2.2])
    assert list(smp.branch_index(turns * 2 * math.pi, step)) == [0, 1, 0, -1, 2]


@pytest.mark.parametrize(
    'step, verdict, branches',
    [
        (UQ_STEP, StepVerdict.ALIASED_UNAMBIGUOUS, (1,)),  # 0.984-1.014 turns per step
        (1.0, StepVerdict.RESOLVED, (0,)),  # <= 73 deg per step
        (2.0, StepVerdict.MARGINAL, (0,)),  # 142-146 deg per step
        (2.5, StepVerdict.AMBIGUOUS, (0, 1)),  # 0.492-0.507 turn: the "forbidden" step of NUMBERS §2.9
        (10.0, StepVerdict.ALIASED_UNAMBIGUOUS, (2,)),
    ],
)
def test_step_verdict_of_1999_uq(step, verdict, branches):
    assert smp.step_verdict(*UQ_RATE_RANGE, step, 90 * DEG) == (verdict, branches)


def test_branch_margin_turns():
    # 0.30..0.35 turn per step: 0.15 to the half-turn; a range across 0.5 has no margin
    assert smp.branch_margin_turns(0.30 * 2 * math.pi, 0.35 * 2 * math.pi, 1.0) == pytest.approx(0.15)
    assert smp.branch_margin_turns(0.45 * 2 * math.pi, 0.55 * 2 * math.pi, 1.0) == 0.0
    # 1999 UQ at 5.0012 yr: 1 - 0.984 = 0.016 below and 1.014 above -> 0.5 - 0.0159 = 0.484 turn
    assert smp.branch_margin_turns(*UQ_RATE_RANGE, UQ_STEP) == pytest.approx(0.484, abs=1e-3)


def test_nyquist_halfwidth_and_ghost_of_1999_uq():
    # a^2.5 / (3 |k| dt) = 1.0941798^2.5 / (27 x 5.0012222) = 0.0092743 au
    assert smp.nyquist_halfwidth(UQ_A, -9, UQ_STEP) == pytest.approx(0.0092743, rel=1e-4)
    ghost = smp.ghost_axis(UQ_SPEC, TABLE_N, 0.0, UQ_STEP, 1)
    assert ghost == pytest.approx(1.0941756, abs=2e-7)
    # the ghost is where the rate is exactly one turn per step
    assert ctx.expected_rate(UQ_SPEC, ghost, TABLE_N, 0.0) * UQ_STEP == pytest.approx(2 * math.pi, rel=1e-12)
    assert smp.ghost_axis(UQ_SPEC, TABLE_N, 0.0, UQ_STEP, 0) == pytest.approx(ctx.exact_axis(UQ_SPEC, TABLE_N, 0.0))


def test_branch_distances():
    c = ctx.identity_coefficient(-9, UQ_A)
    boundary, spacing = smp.branch_distances(UQ_RATE, c, UQ_STEP)
    # neighbouring branches are 2 delta a_N apart; the boundary is half a branch from its centre
    assert spacing == pytest.approx(2 * smp.nyquist_halfwidth(UQ_A, -9, UQ_STEP), rel=1e-6)
    assert boundary == pytest.approx(0.5 * spacing - abs(UQ_RATE - 2 * math.pi / UQ_STEP) / abs(c), rel=1e-6)


# --- the predictive unwrap ---------------------------------------------------------------------------------


def _uq_like_record(step, span=20000.0):
    t = np.arange(0.0, span + step / 2, step)
    return t, np.mod(UQ_RATE * t + 0.4, 2 * math.pi)


def test_predictive_unwrap_recovers_3999_turns_where_np_unwrap_sees_none():
    t, wrapped = _uq_like_record(UQ_STEP)
    unwrapped, residual = smp.predictive_unwrap(t, wrapped, np.full(len(t) - 1, UQ_RATE + 0.3 * DEG))
    truth = UQ_RATE * (t[-1] - t[0]) / (2 * math.pi)
    assert smp.revolutions(unwrapped) == pytest.approx(truth, abs=1e-6)
    assert round(smp.revolutions(unwrapped)) == 3999
    assert abs(smp.revolutions(np.unwrap(wrapped))) < 0.3  # -0.0039 deg/yr x 20 kyr = -0.22 turn
    assert np.max(np.abs(residual)) == pytest.approx(0.3 * DEG * UQ_STEP, rel=1e-6)
    # the 1-yr record needs no model: both unwraps count 3999
    t1, wrapped1 = _uq_like_record(1.0)
    assert round(smp.revolutions(np.unwrap(wrapped1))) == 3999
    assert round(smp.revolutions(smp.predictive_unwrap(t1, wrapped1, UQ_RATE)[0])) == 3999


def test_a_230_degree_model_error_reads_as_minus_130_and_adds_a_wrong_turn():
    t = np.array([0.0, 1.0])
    wrapped = np.array([0.0, np.mod(230 * DEG, 2 * math.pi)])
    unwrapped, residual = smp.predictive_unwrap(t, wrapped, 0.0)
    assert residual[0] / DEG == pytest.approx(-130.0)
    assert (unwrapped[1] - unwrapped[0]) / DEG == pytest.approx(-130.0)  # truth +230: one turn lost


def test_unwrap_planet_longitude_at_five_turns_per_step():
    t = np.arange(0.0, 1000.0, UQ_STEP)
    longitude = TABLE_N['Earth'] * t
    unwrapped = smp.unwrap_planet_longitude(t, np.mod(longitude, 2 * math.pi), TABLE_N['Earth'])
    assert unwrapped - unwrapped[0] == pytest.approx(longitude - longitude[0], abs=1e-9)


def test_rate_model_uses_midpoint_axes():
    a = np.array([3.0, 3.2, 3.1])
    spec = ResonanceSpec.from_key('2J-1')
    expected = [2 * TABLE_N['Jupiter'] - float(ctx.mean_motion(m)) + 1e-5 for m in (3.1, 3.15)]
    assert smp.rate_model(spec, a, TABLE_N, 1e-5) == pytest.approx(expected)


# --- jitter, safe steps, resampling, aliased lines ----------------------------------------------------------


def test_perihelion_jitter_from_short_period_e():
    # e = 0.02 + 0.002 sin(2 pi t / 5 yr): de_sp = 0.002 / sqrt(2), |p| = 10 -> 10 x 0.0014142 / 0.02 = 0.7071
    t = np.arange(0.0, 5000.0, 0.25)
    e = 0.02 + 0.002 * np.sin(2 * math.pi * t / 5.0) + 0.001 * np.sin(2 * math.pi * t / 2000.0)
    assert smp.perihelion_jitter(t, e, 10, 50.0, 2) == pytest.approx(0.7071, rel=0.02)
    assert smp.perihelion_jitter(t, e, 0, 50.0, 2) == 0.0
    # band below the Nyquist period: every sampled variation is short-period (white-noise estimate)
    rng = np.random.default_rng(1)
    t100 = np.arange(0.0, 100000.0, 100.0)
    noise = 0.02 + 0.001 * rng.standard_normal(len(t100))
    assert smp.perihelion_jitter(t100, noise, 1, 50.0, 2) == pytest.approx(0.05, rel=0.05)


def test_safe_steps_by_hand():
    # plain: 0.5 pi / 72 deg/yr = 1.25 yr; predictive: 0.5 x 1.0942^2.5 / (27 x 1.4e-4) = 165.6 yr (SAMPLING §4: 163)
    plain, predictive = smp.safe_steps(70 * DEG, -72 * DEG, 1.0942, -9, 1.4e-4, 0.5)
    assert plain == pytest.approx(1.25)
    assert predictive == pytest.approx(0.5 * 1.0942**2.5 / (27 * 1.4e-4))
    assert smp.safe_steps(0.0, 0.0, 2.0, 0, 1e-3, 0.5) == (math.inf, math.inf)


def test_resample_step():
    settings = settings_from_config()
    inf, nan = math.inf, float('nan')
    assert smp.resample_step(inf, 7.3, 300.0, 1000.0, settings) == 5.0  # min(7.3, 300 / 20 = 15) -> 5
    assert smp.resample_step(inf, 400.0, 30.0, 1000.0, settings) == 1.0  # 30 / 20 = 1.5 -> 1
    assert smp.resample_step(inf, 400.0, nan, 1000.0, settings) == 10.0
    assert smp.resample_step(inf, 0.03, 300.0, 1000.0, settings) is None
    assert smp.resample_step(1.3, 400.0, 300.0, 1000.0, settings) == 1.0  # the plain (alias) bound: 1.3 -> 1


def test_resample_step_is_finer_than_the_record():
    # 99942 7E-6 at 0.2 yr: max|nu| = 2 pi / 0.2 x 0.6 turns = 18.85 rad/yr (ambiguous: branches 0 and 1), a smooth a
    # (safe_pred 400 yr). The alias needs |nu| dt <= 90 deg: dt <= 0.5 pi / 18.85 = 0.0833 yr -> 0.05, never 10 yr
    settings = settings_from_config()
    plain, pred = smp.safe_steps(-2.0, 0.6 * 2 * math.pi / 0.2, 1.0, -6, 1e-9, 0.5)
    assert plain == pytest.approx(0.5 * math.pi / (0.6 * 2 * math.pi / 0.2))
    assert smp.resample_step(plain, pred, float('nan'), 0.2, settings) == 0.05
    # nothing finer than the record's own step: no recommendation, even when a coarser one would be 'safe'
    assert smp.resample_step(math.inf, 400.0, float('nan'), 0.05, settings) is None
    # the record's step already meets every bound (354652 53J-20 at 0.6 yr, unwrap failed): a finer one cures nothing
    assert smp.resample_step(math.inf, 400.0, float('nan'), 0.6, settings) is None
    assert smp.resample_step(3.0, 400.0, float('nan'), 2.0, settings) is None
    assert smp.resample_step(3.0, 400.0, float('nan'), 10.0, settings) == 2.5
    assert smp.resample_step(0.04, 400.0, float('nan'), 0.2, settings) is None  # below the finest round value


def test_aliased_lines_fold_into_the_libration_band():
    # 1/P_syn = 1 + 1/500 per yr at a 1-yr step folds to 1/500 per yr: a 500-yr line
    lines = smp.aliased_lines({'Jupiter': 1 / (1 + 1 / 500)}, 1.0, 400.0, 1)
    assert lines == pytest.approx((500.0,))
    assert smp.aliased_lines({'Jupiter': 1 / (1 + 1 / 500)}, 1.0, 2000.0, 1) == ()  # outside [1000, 4000]
    assert smp.aliased_lines({'Jupiter': 11.86}, 1.0, float('nan'), 3) == ()


# --- whole record ---------------------------------------------------------------------------------------------


def _uq_series(step, with_planets=True, span=4000.0, noise=1e-4):
    """A 1999 UQ-like circulation of 7E+2M-9: a = <a> + short-period noise, sigma from the longitudes."""
    t = np.arange(0.0, span + step / 2, step)
    rng = np.random.default_rng(7)
    a = UQ_A + noise * np.sin(2 * math.pi * t / 1.27) + 0.2 * noise * rng.standard_normal(len(t))
    n = float(ctx.mean_motion(UQ_A))
    body_lambda = n * t + 1.0
    lambdas = {'Earth': TABLE_N['Earth'] * t + 2.0, 'Mars': TABLE_N['Mars'] * t + 3.0}
    sigma = 7 * lambdas['Earth'] + 2 * lambdas['Mars'] - 9 * body_lambda
    ones = np.ones_like(t)
    planets = {}
    if with_planets:
        for name, longitude in lambdas.items():
            planets[name] = PlanetSeries(
                name, t, ones * const.PLANETS_AXIS[name], 0.03 * ones, 0.01 * ones, ones, ones, np.mod(longitude, 2 * math.pi)
            )
    return MMRSeries(
        t, a, 0.0157 * ones, 0.43 * ones, -2.6 * ones, 0.13 * ones, -2.47 * ones, np.mod(sigma, 2 * math.pi), UQ_SPEC,
        Regime(step, span, len(t)), planets, np.mod(body_lambda, 2 * math.pi), name='uq-like',
    ), (7 * TABLE_N['Earth'] + 2 * TABLE_N['Mars'] - 9 * n)  # fmt: skip


@pytest.mark.parametrize('with_planets', [True, False])
def test_analyse_sampling_on_a_uq_like_record_at_the_5_yr_step(with_planets):
    series, true_rate = _uq_series(UQ_STEP, with_planets)
    series.validate()
    settings = settings_from_config()
    context = ctx.build_context(series, settings)
    sampling = smp.analyse_sampling(series, context, settings)
    assert sampling.verdict == StepVerdict.ALIASED_UNAMBIGUOUS and sampling.branches == (1,)
    assert sampling.unwrap_ok and sampling.unwrap_resid_max_rad < 10 * DEG
    truth = true_rate * (series.t_yr[-1] - series.t_yr[0]) / (2 * math.pi)
    assert sampling.revs_predictive == pytest.approx(truth, abs=0.01)
    assert sampling.alias_numpy_differs and 'alias_np_unwrap_differs' in sampling.flags
    assert context.rate_measured_rad_yr == pytest.approx(true_rate, rel=1e-6)
    assert sampling.far_from_resonance  # 0.018 au from a*, far outside the a range of the record
    assert sampling.rate_apparent_from_mean_longitude_rad_yr == pytest.approx(true_rate - 2 * math.pi / UQ_STEP, abs=1e-8)
    assert sampling.nyquist_rad_yr == pytest.approx(math.pi / UQ_STEP)
    if with_planets:
        assert sampling.angle_reconstruction_rad < 1e-9
    else:
        assert math.isnan(sampling.angle_reconstruction_rad)
    columns = sampling.to_columns()
    assert columns['smp_verdict'] == 'aliased_unambiguous' and columns['smp_branches'] == '1'
    assert columns['smp_nyquist_deg_yr'] == pytest.approx(180 / UQ_STEP)
    assert columns['smp_aliased_lines_yr'] == ''


def test_numpy_mode_keeps_the_residual_but_draws_np_unwrap():
    series, true_rate = _uq_series(UQ_STEP)
    settings = settings_from_config(mmr_unwrap='numpy')
    context = ctx.build_context(series, settings)
    sampling = smp.analyse_sampling(series, context, settings)
    assert sampling.unwrap_ok and sampling.unwrap_method == 'numpy'
    assert smp.revolutions(sampling.angle_unwrapped) == pytest.approx(sampling.revs_numpy)
    # np.unwrap draws the aliased rate: true - 2 pi / dt
    assert context.rate_measured_rad_yr == pytest.approx(true_rate - 2 * math.pi / UQ_STEP, abs=1e-7)


def test_unstitchable_record_fails_the_residual_check():
    series, _ = _uq_series(UQ_STEP, noise=2e-2)  # a noise of 0.02 au: model errors of ~ +-pi per step
    settings = settings_from_config()
    sampling = smp.analyse_sampling(series, ctx.build_context(series, settings), settings)
    assert not sampling.unwrap_ok
    assert sampling.suspect_times_yr and 'unwrap_step_suspect' in sampling.flags


def test_angle_reconstruction_error_sees_an_inconsistent_point():
    series, _ = _uq_series(1.0, span=200.0)
    broken = series.angle_wrapped.copy()
    broken[50] = np.mod(broken[50] + 0.1, 2 * math.pi)
    error = smp.angle_reconstruction_error(MMRSeries(**{**series.__dict__, 'angle_wrapped': broken}))
    assert error == pytest.approx(0.1, rel=0.02)


# --- real 1999 UQ records (benchmark data, local only: marked benchmark) -------------------------------------

UQ_5YR = 'mmr3__1999_UQ__7E+2M-9+0+0+0'
UQ_1YR = 'mmr3__1999_UQ__7E+2M-9+0+0+0__step1yr'


def _load_example(directory):
    import pandas as pd

    frame = pd.read_csv(directory / 'series.csv.gz')
    planets = {}
    if (directory / 'planets.csv.gz').exists():
        saved = pd.read_csv(directory / 'planets.csv.gz')
        for name in ('Earth', 'Mars'):
            columns = [saved[f'{name}.{x}'].values for x in ('a', 'e', 'inc', 'Omega', 'varpi', 'mean_longitude')]
            planets[name] = PlanetSeries(name, saved.t_yr.values, *columns)
    t = frame.t_yr.values
    return MMRSeries(
        t, frame.a.values, frame.e.values, frame.inc.values, frame.Omega.values, frame.omega.values, frame.varpi.values,
        np.mod(frame.angle_raw.values, 2 * math.pi), UQ_SPEC, Regime(float(np.median(np.diff(t))), t[-1] - t[0], len(t)),
        planets, frame.mean_longitude.values, frame.M.values, frame.angle_raw.values, directory.name,
    )  # fmt: skip


@pytest.mark.benchmark
def test_real_1999_uq_alias_at_5_yr_and_3999_turns_at_1_yr():
    settings = settings_from_config()
    results = {}
    for example_id in (UQ_5YR, UQ_1YR):
        series = _load_example(benchmark_data() / 'examples' / example_id)
        context = ctx.build_context(series, settings)
        results[example_id] = (context, smp.analyse_sampling(series, context, settings))
    context5, coarse = results[UQ_5YR]
    context1, fine = results[UQ_1YR]
    assert coarse.verdict == StepVerdict.ALIASED_UNAMBIGUOUS and coarse.branches == (1,) and coarse.unwrap_ok
    assert fine.verdict == StepVerdict.RESOLVED and fine.branches == (0,)
    assert round(fine.revs_predictive) == 3999 and round(fine.revs_numpy) == 3999
    # the 5-yr record lacks 0.11 yr of span: 3998.84 against 3998.86
    assert coarse.revs_predictive == pytest.approx(fine.revs_predictive, abs=0.05)
    assert abs(coarse.revs_numpy) < 0.5
    assert coarse.rate_apparent_meas_rad_yr / DEG == pytest.approx(-0.00382, abs=2e-4)
    assert coarse.rate_apparent_from_mean_longitude_rad_yr / DEG == pytest.approx(-0.00387, abs=2e-4)
    assert coarse.ghost_distance_au < 1e-5 and coarse.far_from_resonance
    assert context1.planet_mean_motion_source == {'Earth': 'measured', 'Mars': 'measured'}
    assert fine.angle_reconstruction_rad < 1e-6
