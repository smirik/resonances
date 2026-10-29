"""W3 guards, F4: a record outside the MMR model gets an invalid verdict with its reason, never an exception.

Counterexamples of Codex astra (PHYSICS-MATH §7): a < 0 raised TypeError (complex n), e = 0 and k = 0 raised
ZeroDivisionError, NaN inc and e < 0 went through; `MMRSeries.validate` accepted planet times shifted by half a step.
The record is a small pendulum libration of 2J-1 (2 kyr, 1-yr step) so that every case is fast.
"""

import dataclasses

import numpy as np
import pytest

from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.models import MMRSeries, PlanetSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import MMR_CHECKS, Action, CheckValue, Reason, Status

S = settings_from_config()
N = 2001


def pendulum_series(n=N, step=1.0) -> MMRSeries:
    """sigma = 1 rad x cos(2 pi t / 1000) about 0 and a answering it (a - a* = sigma_dot / c) near 2:1 with Jupiter."""
    t = np.arange(n) * step
    n_jupiter = np.radians(30.3496)  # rad / Gaussian yr
    a_star = (np.pi / n_jupiter) ** (2 / 3)  # n = 2 n_J
    c = 3 * np.pi * a_star**-2.5
    omega = 2 * np.pi / 1000.0
    sigma = np.cos(omega * t)
    a = a_star - omega * np.sin(omega * t) / c
    zero = np.zeros(n)
    return MMRSeries(
        t, a, np.full(n, 0.1), zero, zero, zero, zero, sigma % (2 * np.pi), ResonanceSpec.from_key('2J-1'), Regime(step, t[-1], n)
    )


def changed(**fields) -> MMRSeries:
    return dataclasses.replace(pendulum_series(), **fields)


def with_one(name: str, value: float, index: int = N // 2) -> MMRSeries:
    series = pendulum_series()
    values = np.array(getattr(series, name), dtype=float)
    values[index] = value
    return dataclasses.replace(series, **{name: values})


@pytest.mark.parametrize(
    'series, problem, reason',
    [
        (changed(a=-pendulum_series().a), 'a<=0', Reason.UNPHYSICAL),  # was TypeError (complex mean motion)
        (with_one('a', 0.0), 'a<=0', Reason.UNPHYSICAL),
        (with_one('e', -0.01), 'e<0', Reason.UNPHYSICAL),
        (with_one('e', 1.0), 'e>=1', Reason.UNPHYSICAL),  # a parabola: n(a) and the periodic angle are undefined
        (with_one('e', 1.2), 'e>=1', Reason.UNPHYSICAL),  # below the package's 1.3, still no ellipse
        (with_one('inc', np.nan), 'non_finite:inc', Reason.UNPHYSICAL),  # went through before
        (with_one('angle_wrapped', np.inf), 'non_finite:angle_wrapped', Reason.UNPHYSICAL),
        (changed(resonance=ResonanceSpec.from_key('1J+0')), 'k=0', Reason.INVALID_INPUT),  # was ZeroDivisionError
    ],
)
def test_a_record_outside_the_model_is_invalid_with_its_reason(series, problem, reason):
    assert problem in series.domain_problems()
    analysis = analyse_mmr(series, S)
    verdict = analysis.verdict
    assert (verdict.status, verdict.reason, verdict.action) == (Status.INVALID, reason, Action.REVIEW)
    assert all(verdict.checks[name] == CheckValue.NOT_MEASURED for name in MMR_CHECKS)
    numbers, summary = analysis.numbers_row(), analysis.summary_row()
    assert problem in numbers['rec_input_problems'].split(';')
    assert summary['status'] == 'invalid' and summary['status_code'] == -99
    assert [entry.column for entry in verdict.decision_path] == ['rec_body_coefficient', 'rec_unphysical']
    analysis.drop_arrays()  # no sampling to release: must not fail


@pytest.mark.parametrize('seed', range(5))
def test_the_package_unphysical_rule_is_inside_the_domain_check(seed):
    """Why G0 has no unphysical branch for an MMR any more: whatever `classify.util.unphysical_arrays` flags (non-finite
    a or e, e > 1.3, a < 0), and a non-finite angle or time, `domain_problems` already refuses (non-finite, e >= 1,
    a <= 0). Values straddle both thresholds: e 1.0 and 1.2 are refused by the domain check only, 1.31 by both."""
    from resonances.resonance.classify.util import unphysical_arrays

    rng = np.random.default_rng(seed)
    values_a = np.array([-1.0, 0.0, 1e-9, 2.5, 3.3, np.nan, np.inf])
    values_e = np.array([-0.01, 0.0, 0.5, 0.999, 1.0, 1.2, 1.3, 1.31, 2.0, np.nan, -np.inf])
    flagged = 0
    for _ in range(200):
        series = pendulum_series(n=50)
        a = np.array(series.a, dtype=float)
        e = np.array(series.e, dtype=float)
        angle = np.array(series.angle_wrapped, dtype=float)
        a[rng.integers(50)] = rng.choice(values_a)
        e[rng.integers(50)] = rng.choice(values_e)
        if rng.random() < 0.2:
            angle[rng.integers(50)] = np.nan
        series = dataclasses.replace(series, a=a, e=e, angle_wrapped=angle)
        if unphysical_arrays(a, e)[0] or not np.all(np.isfinite(angle)):
            flagged += 1
            assert series.domain_problems(), (a[np.isfinite(a) & (a <= 3.3)], e)
    assert flagged > 50  # the draw hits the flagged side often enough to mean something


def test_a_hyperbolic_orbit_is_refused_before_g0():
    """e = 1.5 (the package's 'hyperbolic', e > 1.3) is invalid/unphysical from the input check, with its decision path,
    never undefined from G0."""
    series = with_one('e', 1.5)
    assert series.domain_problems() == ['e>=1']
    verdict = analyse_mmr(series, S).verdict
    assert (verdict.status, verdict.reason) == (Status.INVALID, Reason.UNPHYSICAL)
    assert [entry.column for entry in verdict.decision_path] == ['rec_body_coefficient', 'rec_unphysical']


def test_every_problem_is_listed():
    series = changed(a=np.full(N, -1.0), e=np.full(N, -0.1), resonance=ResonanceSpec.from_key('1J+0'))
    assert series.domain_problems() == ['k=0', 'a<=0', 'e<0']
    assert pendulum_series().domain_problems() == []


def test_a_circular_orbit_is_inside_the_model():
    """e = 0 exactly is an ellipse; the analysis gives a verdict (ZeroDivisionError before, Codex astra §7)."""
    series = changed(e=np.zeros(N))
    assert series.domain_problems() == []
    verdict = analyse_mmr(series, S, scan_owner=False, scan_subresonances=False).verdict
    assert verdict.status != Status.INVALID


def _with_planet(times: np.ndarray) -> MMRSeries:
    series = pendulum_series()
    zero = np.zeros(N)
    jupiter = PlanetSeries('Jupiter', times, zero + 5.2, zero + 0.05, zero, zero, zero, 0.53 * times)
    return dataclasses.replace(series, planets={'Jupiter': jupiter})


def test_planet_times_must_be_the_record_times():
    t = pendulum_series().t_yr
    _with_planet(t + 1e-4).validate()  # 1e-4 of a step: CSV rounding, accepted
    with pytest.raises(ValueError, match='Jupiter'):
        _with_planet(t + 0.5).validate()  # the same length, half a step later: refused (accepted before)
