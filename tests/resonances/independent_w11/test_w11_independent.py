"""Independent functional tests of W11: the witness of rule R needs an identity that does not fail, and every regime
between confirmed jumps of a selects its turning points at its own prominence.

Expected values are derived here, not read from the implementation:

- witness (impl/API.md, W11 witness): a strict L4 PASS is a witness when B1 does not fail and L3 passes or is not
  measured; a measured L3 FAIL (a and sigma follow another clock) or not_applicable makes none; a thin L4 PASS needs
  L3 and B1 passing; a pooled measurement (no episode id) is never a witness;
- synthetic 2J-1 records: a pendulum sigma_ddot = -w0^2 sin(sigma - pi), w0 = 2 pi / 600 yr, half-amplitude 90 deg,
  with the body's n = 2 n_J - sigma_dot and a from Kepler's third law. Its period is
  T = (2/pi) K(m = sin^2 45 deg) T0 = (2/pi) 1.85407 x 600 = 708.2 yr; started at the centre with a positive rate,
  its turning points sit at T/4 + n T/2 = 177.0 + 354.1 n yr, so a libration on [0, 10 kyr) has 28 of them
  (n = 0..27, the last at 9738 yr); the next one would come after the libration has ended;
- a turning point of a libration of half-amplitude 90 deg has prominence at most 2 x 90 = 180 deg, so a prominence
  threshold above 180 deg leaves none of them;
- after 10 kyr sigma circulates in segments of 250 yr whose direction alternates (periods 25-55 yr, faster than the
  shortest libration of 150 yr): each reversal is a step of a by several per cent, and the triangle wave of the unwrapped
  sigma is what sigma_f holds faster than 150 yr, tens of degrees at the 95th percentile over the record.
"""

import dataclasses
import math
from functools import lru_cache

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from resonances.data import const
from resonances.resonance.analysis import geometry as geometry_module
from resonances.resonance.analysis.mmr import response as rsp
from resonances.resonance.analysis.mmr.analyse import analyse_mmr, fast_residual
from resonances.resonance.analysis.mmr.models import EpisodeResponse, MMRSeries, Regime, ResonanceSpec, Response
from resonances.resonance.analysis.mmr.models import ResponseSource
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import CheckValue

SETTINGS = AnalysisSettings()
PASS, FAIL, NM, NA = CheckValue.PASS, CheckValue.FAIL, CheckValue.NOT_MEASURED, CheckValue.NOT_APPLICABLE

# ------------------------------------------------------------------------------------------- the witness (L3)

_EMPTY = {
    'float': lambda: float('nan'),
    'int': lambda: 0,
    'bool': lambda: False,
    'str': lambda: '',
    'CheckValue': lambda: CheckValue.NOT_MEASURED,
    'np.ndarray': lambda: np.array([]),
}


def blank(cls, **values):
    """An instance with every required field not in `values` set to 'not measured' (nan, None, empty)."""
    kwargs = dict(values)
    for f in dataclasses.fields(cls):
        has_default = f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
        if f.name not in values and not has_default:
            kind = str(f.type)
            if kind.startswith('List'):
                kwargs[f.name] = []
            elif kind.startswith('Tuple'):
                kwargs[f.name] = ()
            else:
                kwargs[f.name] = _EMPTY.get(kind, lambda: None)()
    return cls(**kwargs)


def judged(episode_id=0, l4=PASS, thin=False, l3=PASS, b1=PASS, n_tp=20, r2=0.9, slope=1.0):
    return blank(
        EpisodeResponse,
        episode_id=episode_id,
        before_jump=False,
        n_tp=n_tp,
        t_start_yr=0.0,
        t_end_yr=1000.0,
        period_yr=600.0,
        force_r2_h1=r2,
        force_r2_h2=r2,
        identity_slope_ratio=slope,
        identity_corr=0.9,
        b1_ratio=1.0,
        a_response=l4,
        a_response_thin=thin,
        identity=l3,
        a_amplitude=b1,
    )


def pooled(episodes, l4=PASS, l3=PASS, b1=PASS):
    return blank(
        Response,
        a_response=l4,
        a_response_thin=False,
        identity=l3,
        a_amplitude=b1,
        force_r2_h1=0.4,
        force_r2_h2=0.4,
        identity_slope_ratio=0.7,
        identity_corr=0.5,
        b1_ratio=3.0,
        slow_libration=FAIL,
        episode_responses=list(episodes),
        before_jump_response=None,
    )


@pytest.mark.parametrize(
    'l4, thin, l3, b1, expected',
    [
        # strict L4: L3 must not be measured to fail, B1 must not fail
        (PASS, False, PASS, PASS, True),
        (PASS, False, NM, PASS, True),
        (PASS, False, FAIL, PASS, False),  # the W11 change: was a witness under W10
        (PASS, False, NA, PASS, False),  # not_applicable is not "pass or not_measured"
        (PASS, False, PASS, NM, True),
        (PASS, False, NM, NM, True),
        (PASS, False, FAIL, NM, False),
        (PASS, False, PASS, FAIL, False),
        (PASS, False, NM, FAIL, False),
        (PASS, False, FAIL, FAIL, False),
        # thin L4: L3 and B1 must pass
        (PASS, True, PASS, PASS, True),
        (PASS, True, NM, PASS, False),
        (PASS, True, FAIL, PASS, False),
        (PASS, True, PASS, NM, False),
        # no L4 pass: never
        (FAIL, False, PASS, PASS, False),
        (NM, False, PASS, PASS, False),
        (NA, False, PASS, PASS, False),
    ],
)
def test_witness_truth_table(l4, thin, l3, b1, expected):
    assert judged(l4=l4, thin=thin, l3=l3, b1=b1).witness is expected


@pytest.mark.parametrize('l3', [PASS, NM, FAIL])
def test_a_pooled_measurement_is_never_a_witness(l3):
    pool = dataclasses.replace(judged(l3=l3), episode_id=None, before_jump=True)
    assert pool.witness is False


def test_episode_with_failed_identity_does_not_decide_the_checks():
    """Strict L4 with L3 FAIL is no witness: the record's pooled response stays, numbers and sources unchanged."""
    other_clock = judged(0, n_tp=40, r2=0.95, l3=FAIL, slope=-1.4)
    response = pooled([other_clock], l4=FAIL, l3=PASS)
    assert rsp.witnesses(response) == []
    d = rsp.deciding_response(response, before_jump=False)
    assert d.witness_episode_id is None
    assert (d.force_source, d.identity_source, d.b1_source) == (ResponseSource.RECORD,) * 3
    assert d.a_response == FAIL and d.force_r2_h1 == 0.4 and d.identity_slope_ratio == 0.7


def test_witness_with_unmeasured_identity_decides_the_force_law():
    response = pooled([judged(0, n_tp=12, r2=0.93, l3=NM, slope=float('nan'))], l4=FAIL, l3=PASS)
    d = rsp.deciding_response(response, before_jump=False)
    assert d.witness_episode_id == 0
    assert d.force_source == ResponseSource.WITNESS and d.a_response == PASS and d.force_r2_h1 == 0.93
    # its not-measured identity never replaces the measured pooled one
    assert d.identity == PASS and d.identity_source == ResponseSource.RECORD and d.identity_slope_ratio == 0.7


def test_failed_identity_episode_is_skipped_in_favour_of_a_shorter_witness():
    """The witness with most TPs decides; an L3-FAIL episode with more TPs is not a candidate at all."""
    other_clock = judged(0, n_tp=40, r2=0.95, l3=FAIL, slope=-1.3)
    genuine = judged(1, n_tp=10, r2=0.88, l3=PASS, slope=1.02)
    response = pooled([other_clock, genuine], l4=FAIL, l3=FAIL)
    assert [w.episode_id for w in rsp.witnesses(response)] == [1]
    d = rsp.deciding_response(response, before_jump=False)
    assert d.witness_episode_id == 1
    assert d.force_r2_h1 == 0.88 and d.identity == PASS and d.identity_slope_ratio == 1.02
    assert d.to_columns()['rsp_n_witnesses'] == 1
    # the L3-FAIL episode's force law passes: no conflict with the witness
    assert not rsp.force_law_mixed(response)


def test_no_conflict_flag_without_a_witness():
    """An L3-FAIL episode next to an L4-FAIL one: no witness, so nothing for the failed force law to conflict with."""
    response = pooled([judged(0, l3=FAIL), judged(1, l4=FAIL)])
    assert rsp.witnesses(response) == []
    assert not rsp.force_law_mixed(response)


# ------------------------------------------------------------------------------------------- synthetic records

N_J = math.radians(const.PLANET_N_DEG_PER_GAUSS_YR['Jupiter'])
W0 = 2 * np.pi / 600.0
FINE_YR = 0.5
SPAN_YR = 20000.0
LIBRATION_END_YR = 10000.0
HALF_AMP_DEG = 90.0
PERIOD_YR = (2 / np.pi) * 1.854075 * 600.0  # K(m = 1/2) = 1.854075
EXPECTED_TP_YR = PERIOD_YR / 4 + np.arange(28) * PERIOD_YR / 2
FAST_PERIODS = (30, 45, 25, 55, 35, 28, 50, 40)
SLOWER_PERIODS = (40, 60, 33, 70, 45, 36, 65, 50)


def _pendulum(t, sigma0, rate0):
    sol = solve_ivp(
        lambda _, y: [y[1], -(W0**2) * np.sin(y[0] - np.pi)],
        (t[0], t[-1]),
        [sigma0, rate0],
        t_eval=t,
        rtol=1e-10,
        atol=1e-12,
        max_step=FINE_YR,
    )
    return sol.y


@lru_cache(maxsize=None)
def trajectory(late):
    """(t, sigma, sigma_dot) on a 0.5-yr grid. `late`: None = libration to the end, else the circulation periods
    [yr] of the 250-yr segments after LIBRATION_END_YR (alternating direction, sigma continuous)."""
    t = np.arange(0.0, SPAN_YR + FINE_YR / 2, FINE_YR)
    sigma, rate = np.empty_like(t), np.empty_like(t)
    end = SPAN_YR + 1 if late is None else LIBRATION_END_YR
    m = t < end
    sigma[m], rate[m] = _pendulum(t[m], np.pi, W0 * 2 * math.sin(math.radians(HALF_AMP_DEG) / 2))
    if late is not None:
        seg, start, i = 250.0, LIBRATION_END_YR, 0
        while start < SPAN_YR - seg / 2:
            stop = start + seg if start + 1.5 * seg < SPAN_YR else SPAN_YR + 1
            m = (t >= start) & (t < stop)
            direction = 1 if i % 2 == 0 else -1
            previous = sigma[np.flatnonzero(m)[0] - 1]
            sigma[m], rate[m] = _pendulum(t[m], previous, direction * 2 * np.pi / late[i % len(late)])
            start, i = start + seg, i + 1
    return t, sigma, rate


def make_series(late, step_yr):
    t, sigma, rate = trajectory(late)
    k = int(round(step_yr / FINE_YR))
    t, sigma, rate = t[::k], sigma[::k], rate[::k]
    varpi = 0.7
    lam = 2 * (0.3 + N_J * t) - varpi - sigma  # sigma = 2 lambda_J - lambda - varpi
    a = (4 * np.pi**2 / (2 * N_J - rate) ** 2) ** (1.0 / 3.0)
    one = np.ones_like(t)
    return MMRSeries(
        t_yr=t,
        a=a,
        e=0.15 * one,
        inc=0.05 * one,
        Omega=1.1 * one,
        omega=(varpi - 1.1) * one,
        varpi=varpi * one,
        angle_wrapped=np.mod(sigma, 2 * np.pi),
        resonance=ResonanceSpec.from_key('2J-1'),
        regime=Regime(step_yr=step_yr, span_yr=float(t[-1]), n_rows=len(t), integrator='ias15'),
        mean_longitude=np.mod(lam, 2 * np.pi),
        mean_anomaly=np.mod(lam - varpi, 2 * np.pi),
        name='w11-independent',
    )


@lru_cache(maxsize=None)
def analysed(late, step_yr):
    return analyse_mmr(make_series(late, step_yr), SETTINGS)


def early_episodes(analysis):
    return [run for run in analysis.geometry.runs if run.accepted and run.t_end_yr < LIBRATION_END_YR]


def confirmed_jumps(analysis):
    return [jump for jump in analysis.geometry.jumps if not jump.kick]


def record_prominence(analysis):
    """p_TP of docs/physics/geometry.md ("Turning points") over the whole record: max(floor, factor x max(ripple, fast residual))."""
    ripple = max(analysis.prep.unremoved_ripple_rad, fast_residual(analysis.prep, SETTINGS))
    return max(math.radians(SETTINGS.mmr_tp_prominence_floor_deg), SETTINGS.mmr_tp_ripple_factor * ripple)


@pytest.mark.xfail(
    strict=True,
    reason=(
        'W16: at tau = 2 P_c = 150 yr the locator joins a-increment stretches less than tau apart into one '
        'group; the reversals are 250 yr apart (< 2 tau), of similar size, and the group (9.7 kyr) is '
        'rejected as a drift: no confirmed jump (TEST-DISPUTES W16)'
    ),
)
def test_synthetic_record_has_many_jumps_and_a_record_prominence_that_hides_the_libration():
    """The premise of the fix: the record-wide threshold is above the 180-deg ceiling of the libration's TPs."""
    analysis = analysed(FAST_PERIODS, 1.0)
    jumps = confirmed_jumps(analysis)
    assert len(jumps) >= 10
    assert all(j.time_yr > LIBRATION_END_YR - 300 for j in jumps)  # no jump inside the libration
    p_record = record_prominence(analysis)
    assert p_record > math.radians(2 * HALF_AMP_DEG)
    # with that threshold sigma_f has fewer than one cycle of TPs before the jumps: no episode could be found
    t = np.asarray(analysis.prep.t_yr)
    maxima, minima = geometry_module.extrema(analysis.prep.angle_f).above(p_record)
    early = [i for i in np.concatenate([maxima, minima]) if t[i] < LIBRATION_END_YR]
    assert len(early) < SETTINGS.libration_min_turning_points


@pytest.mark.xfail(
    strict=True,
    reason=(
        'W16: at tau = 2 P_c = 150 yr the locator joins a-increment stretches less than tau apart into one '
        'group; the reversals are 250 yr apart (< 2 tau), of similar size, and the group (9.7 kyr) is '
        'rejected as a drift: no confirmed jump (TEST-DISPUTES W16)'
    ),
)
def test_long_early_libration_before_many_jumps_is_found():
    analysis = analysed(FAST_PERIODS, 1.0)
    episodes = early_episodes(analysis)
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.before_last_jump
    assert episode.n_tp == len(EXPECTED_TP_YR) == 28
    np.testing.assert_allclose(episode.tp_times_yr, EXPECTED_TP_YR, atol=3.0)
    assert episode.period_med_yr == pytest.approx(PERIOD_YR, rel=0.01)
    assert math.degrees(episode.half_amp_med_rad) == pytest.approx(HALF_AMP_DEG, abs=3.0)
    # selected at its regime's own threshold, far below the record's
    assert episode.tp_prominence_rad < math.radians(10.0) < record_prominence(analysis)
    verdict = analysis.verdict
    assert str(verdict.status) == 'transient'
    assert str(verdict.angle_status) == 'episodic'
    assert str(verdict.final_regime) == 'circulation'


def test_no_run_crosses_a_confirmed_jump():
    analysis = analysed(FAST_PERIODS, 1.0)
    half_tau = analysis.geometry.jump_tau_yr / 2
    for run in analysis.geometry.runs:
        for jump in confirmed_jumps(analysis):
            assert not (run.t_start_yr < jump.time_yr - half_tau and run.t_end_yr > jump.time_yr + half_tau)


def test_early_episode_does_not_depend_on_what_follows_the_jumps():
    fast, slower = early_episodes(analysed(FAST_PERIODS, 1.0)), early_episodes(analysed(SLOWER_PERIODS, 1.0))
    assert len(fast) == len(slower) == 1
    np.testing.assert_allclose(fast[0].tp_times_yr, slower[0].tp_times_yr, atol=1.0)
    assert fast[0].tp_prominence_rad == pytest.approx(slower[0].tp_prominence_rad, rel=0.05)


def _verdict_key(analysis):
    v = analysis.verdict
    return str(v.status), str(v.angle_status), str(v.reason), str(v.confidence)


@pytest.mark.parametrize('steps', [(0.5, 2.0), (1.0, 5.0), (2.0, 10.0)])
def test_pure_libration_verdict_does_not_depend_on_a_resolving_step(steps):
    """A 708-yr libration over 20 kyr is resolved at 0.5-10 yr (P_c = 75 yr at every step, 70+ samples per period).
    At or above half the orbital period P_J / 2 / 2 = 2.97 yr the flag short_period_aliased is set (the record cannot
    know that it holds no short-period terms), and since W20 round D its cap (medium) with it: the record never lifts
    it, only a finer step measures what the folded lines do (MMR_SHORT_PERIOD_CAP_RATIO = 0). Status, word and reason
    do not depend on the step; the confidence does only through that cap."""
    first, second = (analysed(None, step) for step in steps)
    assert _verdict_key(first)[:3] == _verdict_key(second)[:3]
    assert str(first.verdict.status) == 'resonant' and str(first.verdict.angle_status) == 'libration'
    nyquist = 2 * np.pi / (2 * N_J) / 2
    capped = []
    for analysis in (first, second):
        flagged = analysis.sampling.step_yr >= nyquist
        assert ('short_period_aliased' in analysis.verdict.flags) is flagged
        assert ('short_period_aliased' in analysis.verdict.caps) is flagged
        capped.append(flagged)
    assert capped == [steps[0] >= 2.97, steps[1] >= 2.97]
    if not any(capped):
        assert _verdict_key(first) == _verdict_key(second)


@pytest.mark.xfail(
    strict=True,
    reason=(
        'W16: at tau = 2 P_c = 150 yr the locator joins a-increment stretches less than tau apart into one '
        'group; the reversals are 250 yr apart (< 2 tau), of similar size, and the group (9.7 kyr) is '
        'rejected as a drift: no confirmed jump (TEST-DISPUTES W16)'
    ),
)
@pytest.mark.parametrize('steps', [(0.5, 1.0), (1.0, 2.0), (1.0, 2.5)])
def test_jump_record_verdict_and_early_episode_do_not_depend_on_a_resolving_step(steps):
    """Steps up to 2.5 yr resolve both the libration and the fastest circulation (10 samples per 25-yr revolution)."""
    first, second = (analysed(FAST_PERIODS, step) for step in steps)
    assert _verdict_key(first) == _verdict_key(second)
    assert str(first.verdict.final_regime) == str(second.verdict.final_regime) == 'circulation'
    a, b = early_episodes(first), early_episodes(second)
    assert len(a) == len(b) == 1
    assert a[0].n_tp == b[0].n_tp == 28
    np.testing.assert_allclose(a[0].tp_times_yr, b[0].tp_times_yr, atol=3.0)
