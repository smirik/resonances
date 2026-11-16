"""Owner test B3: a lattice ripple is not an owner (the ripple test of `owner.accept`).

A body in 11N-18+0+7 (as 307261) librates about pi with A = 1.2 rad and P = 30 kyr. On top of it the term
phi_o = 3 lambda_N - 5 lambda + 2 varpi (3 x 18 - 5 x 11 = -1) turns at n_N / 18 (one turn per 18 P_N =
2966 yr) and moves sigma by EPS sin(phi_o): the planet's shortest forced term, a wave of a at 2966 yr with a
fixed phase on phi_o. Analysed at P = 2966 yr (the period a cycle count on that ripple gives) 3N-5 fits a_dot
with R^2 ~ 1, a fixed phase and psi ~ 0, but sigma below the band [P/2, 2P] librates with 2 A_SLOW against 2 EPS:
the owner of the ripple, not of the libration. Without the ripple test it would be accepted (checked below).

The same term with no slow libration is sigma's whole motion: then the ripple test passes (385266 16N-27: 3N-5
is its owner, 3 x 27 - 5 x 16 = 1).
"""

import math

import numpy as np
import pytest

from resonances.resonance.analysis.mmr import owner
from resonances.resonance.analysis.mmr.context import build_context, table_mean_motion
from resonances.resonance.analysis.mmr.models import MMRSeries, PlanetSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.mmr.preprocess import preprocess
from resonances.resonance.analysis.mmr.sampling import analyse_sampling
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import CheckValue

S = settings_from_config()
TWO_PI = 2 * math.pi

STEP, SPAN = 10.0, 150000.0
N_NEPTUNE = table_mean_motion('Neptune')
P_RIPPLE = 18 * TWO_PI / N_NEPTUNE  # 2966 yr
P_SLOW, A_SLOW = 30000.0, 1.2
EPS = 0.06  # rad: 3.4 deg, as the ripples of the TNO records (1-6 deg)
VARPI_RATE = TWO_PI / 2.0e6


def _planet(t):
    zeros = np.zeros_like(t)
    return PlanetSeries(
        'Neptune', t, 30.07 + zeros, 0.009 + zeros, 0.03 + zeros, 2.3 + zeros, 0.8 + zeros, np.mod(N_NEPTUNE * t + 0.4, TWO_PI)
    )


def series_with(slow_amplitude: float, eps: float) -> MMRSeries:
    """11N-18+0+7 = pi + slow_amplitude sin(2 pi t / P_SLOW) + eps sin(phi_o); lambda follows, a from n = lambda_dot."""
    t = np.arange(0.0, SPAN + STEP / 2, STEP)
    lambda_n = N_NEPTUNE * t + 0.4
    varpi = 0.5 + VARPI_RATE * t
    lambda_0 = (11 * lambda_n + 7 * varpi - math.pi) / 18  # the exact commensurability
    phi_o = 3 * lambda_n - 5 * lambda_0 + 2 * varpi  # turns at n_N / 18
    phi_rate = 3 * N_NEPTUNE - 5 * (11 * N_NEPTUNE + 7 * VARPI_RATE) / 18 + 2 * VARPI_RATE
    w_slow = TWO_PI / P_SLOW
    sigma = math.pi + slow_amplitude * np.sin(w_slow * t) + eps * np.sin(phi_o)
    sigma_dot = slow_amplitude * w_slow * np.cos(w_slow * t) + eps * phi_rate * np.cos(phi_o)
    mean_longitude = (11 * lambda_n + 7 * varpi - sigma) / 18
    n = (11 * N_NEPTUNE + 7 * VARPI_RATE - sigma_dot) / 18
    a = (TWO_PI / n) ** (2 / 3)
    Omega = np.mod(1.0 - TWO_PI / 1.0e6 * t, TWO_PI)
    planets = {'Neptune': _planet(t)}
    series = MMRSeries(
        t_yr=t,
        a=a,
        e=0.1 + np.zeros_like(t),
        inc=0.1 + np.zeros_like(t),
        Omega=Omega,
        omega=np.mod(varpi - Omega, TWO_PI),
        varpi=np.mod(varpi, TWO_PI),
        angle_wrapped=np.mod(sigma, TWO_PI),
        resonance=ResonanceSpec.from_key('11N-18+0+7'),
        regime=Regime(step_yr=STEP, span_yr=SPAN, n_rows=len(t), planets_saved=tuple(planets)),
        planets=planets,
        mean_longitude=np.mod(mean_longitude, TWO_PI),
        name='synthetic-ripple',
    )
    series.validate()
    return series


def _run(series, settings):
    context = build_context(series, settings)
    sampling = analyse_sampling(series, context, settings)
    prep = preprocess(series, context, sampling, settings)
    return owner.run_owner(series, prep, context, P_RIPPLE, settings)


def _three_five(result):
    return next(c for c in result.candidates if c.spec.key == '3N-5+0+2')


@pytest.fixture(scope='module')
def ripple():
    return series_with(A_SLOW, EPS)


def test_lattice_ripple_is_not_an_owner(ripple):
    result = _run(ripple, S)
    assert result.check == CheckValue.PASS and result.accepted is None
    assert owner.FLAG_PROBABLE not in result.flags
    candidate = _three_five(result)
    # every other test passes: the term fits a_dot with a fixed phase and sigma's band is (18/5) phi_o's
    assert candidate.r2_median > 0.9 and candidate.phi0_coherence > 0.95 and candidate.psi_share < 0.05
    assert candidate.reason == owner.REASON_RIPPLE
    if hasattr(candidate, 'sigma_slow_turns'):
        # 150 kyr of a 30-kyr libration: ~10 turning points (fewer after the band-pass edge trim)
        assert candidate.sigma_slow_turns >= 3
        # slow peak-to-peak ~2 A_SLOW against the band's 2 EPS (A_SLOW / EPS = 20)
        assert candidate.sigma_slow_to_band > 10


def test_without_the_ripple_test_the_ripple_would_be_accepted(ripple):
    """A prominence of 1000 x the band's peak-to-peak turns the ripple test off."""
    result = _run(ripple, settings_with_ripple_factor(1000.0))
    assert result.accepted is not None and result.accepted.spec.key == '3N-5+0+2'


@pytest.mark.slow
def test_the_same_term_alone_is_not_a_ripple():
    """No slow libration: sigma's whole motion is the term's. The ripple test passes; at 0.5 rad the identity fits
    a_dot to sigma as well (R^2 ~ 1, fixed phase), so the term is the probable owner, not a proven one. Since W20 the
    potential route measures it too: the synthetic a follows the kinematics, not Neptune's force, so the predicted
    a_dot has the right shape (R^2 passes) and the wrong size (potential_scale_out)."""
    result = _run(series_with(0.0, 0.5), S)
    candidate = _three_five(result)
    assert candidate.rank == 1 and candidate.reason == owner.REASON_POTENTIAL_SCALE
    assert candidate.potential.r2_median >= S.mmr_owner_r2_min
    assert owner.FLAG_PROBABLE in result.flags


def settings_with_ripple_factor(value):
    """mmr_owner_ripple_factor may not be a settings field yet (requests-w2-owner.md): set it either way."""
    try:
        return settings_from_config(mmr_owner_ripple_factor=value)
    except (TypeError, ValueError):
        settings = settings_from_config()
        object.__setattr__(settings, 'mmr_owner_ripple_factor', value)
        return settings
