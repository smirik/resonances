"""Owner test B3 (SPEC §4.8, `analysis/mmr/owner.py`) on hand-computed numbers and synthetic records.

The synthetic body is built from its owner: phi_o = 11 lambda_M - 20 lambda + 9 varpi librates about pi with
A = 0.6 rad and P = 2000 yr, and lambda follows from it: lambda = (11 lambda_M + 9 varpi - phi_o) / 20, a from
n = lambda_dot. The catalogue angle 1V+5J-7+0+0+1 then is sigma = (7/20) phi_o + psi with psi turning once per
15 kyr (Venus' mean motion is chosen so). Expected by construction: the owner is 11M-20+0+9 (libration), and
the angle 11M-20+0+9 itself has no owner.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from resonances.data import const
from resonances.resonance.analysis.mmr import owner
from resonances.resonance.analysis.mmr.context import build_context
from resonances.resonance.analysis.mmr.models import MMRSeries, OwnerCandidate, PlanetSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.mmr.preprocess import preprocess
from resonances.resonance.analysis.mmr.sampling import analyse_sampling
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import CheckValue, OriginDetail

S = settings_from_config()
TWO_PI = 2 * math.pi

# --- notation, candidates, rates (numbers computed by hand) ---------------------------------------------


def test_owner_key_puts_the_eccentricity_on_the_body():
    assert owner.owner_key(('Mars',), (11,), -20, 9) == '11M-20+0+9'
    assert owner.owner_key(('Jupiter',), (3,), -2, -1) == '3J-2+0-1'
    assert owner.owner_key(('Venus', 'Jupiter'), (1, 5), -7, 1) == '1V+5J-7+0+0+1'
    spec = owner.owner_spec(('Earth',), (7,), -23)  # p = -(7 - 23) = 16
    assert spec.key == '7E-23+0+16' and spec.order == 16 and spec.body_varpi_coefficient == 16
    assert owner.owner_spec(('Mars', 'Saturn'), (8, -4), -7).key == '8M-4S-7+0+0+3'


def test_same_lambda_part_is_proportionality_of_the_lambda_coefficients():
    key = ResonanceSpec.from_key
    assert owner.same_lambda_part(key('11M-20+0+9'), key('11M-20+1+8'))  # a subresonance
    assert owner.same_lambda_part(key('11M-20+0+9'), key('22M-40+0+18'))  # a harmonic
    assert owner.same_lambda_part(key('1V+5J-7+0+0+1'), key('2V+10J-14+0+0+2'))
    assert not owner.same_lambda_part(key('10E-17+0+7'), key('11M-20+0+9'))
    assert not owner.same_lambda_part(key('3M+5S-3+0+0-5'), key('10E-17+0+7'))


def test_candidate_rate_by_hand():
    spec = ResonanceSpec.from_key('11M-20+0+9')
    # 11 * 3 - 20 * 1.6 + 9 * 0.001 = 33 - 32 + 0.009
    assert owner.candidate_rate(spec, {'Mars': 3.0}, 1.6, 0.001) == pytest.approx(1.009)


def _context(planet_n: dict, a_lo: float = 2.2, a_hi: float = 2.3):
    """A Context with what enumerate_owners reads (the other fields are not used)."""
    from types import SimpleNamespace

    return SimpleNamespace(
        planet_mean_motion_rad_yr=planet_n, a_p01_au=a_lo, a_p99_au=a_hi, body_varpi_rate_rad_yr=0.0, mean_motion_rad_yr=float('nan')
    )


def test_enumerate_owners_keeps_the_exact_commensurability_only():
    n_mars = 3.3405
    body_n = 11 * n_mars / 20  # 11M-20 exact: k1 n_M - k2 n = 0
    spec = ResonanceSpec.from_key('1V+5J-7+0+0+1')
    # bound 1.5 * 2 pi / 1000 = 0.0094 rad/yr; the next Mars ratio k1/k2 near 0.55 with k1 <= 60 misses it by
    # |k1 - 0.55 k2| n_M >= 0.05 n_M (k2 <= 80): only 11M-20 (22M-40, 33M-60 are harmonics, gcd > 1)
    found = owner.enumerate_owners(
        spec, _context({'Mars': n_mars}), 1000.0, ('Mars',), S, body_n_range=(body_n, body_n), body_varpi_rate=0.0
    )
    assert [c.key for c in found] == ['11M-20+0+9']
    # the angle's own lambda part is never a candidate
    own = ResonanceSpec.from_key('11M-20+0+9')
    assert owner.enumerate_owners(own, _context({'Mars': n_mars}), 1000.0, ('Mars',), S, body_n_range=(body_n, body_n)) == []
    # a planet without a saved longitude does not qualify
    assert owner.enumerate_owners(spec, _context({'Mars': n_mars}), 1000.0, (), S, body_n_range=(body_n, body_n)) == []


def test_enumerate_owners_reads_the_a_range():
    n_earth = const.PLANET_N_DEG_PER_GAUSS_YR['Earth'] * math.pi / 180
    exact = 7 * n_earth / 23  # 7E-23: 7 n_E - 23 n = 0 (varpi rate 0)
    spec = ResonanceSpec.from_key('2R-8E-1+0+0+7')
    context = _context({'Earth': n_earth, 'Mercury': 26.0})
    keys = lambda found: [c.key for c in found]  # noqa: E731
    # the range crosses the commensurability: in, although the middle of the range is 23 * 0.01 = 0.23 rad/yr away
    inside = owner.enumerate_owners(spec, context, 300.0, ('Earth',), S, body_n_range=(exact - 0.01, exact + 0.03), body_varpi_rate=0.0)
    assert '7E-23+0+16' in keys(inside)
    # the range stops 0.005 rad/yr short: |rate| >= 23 * 0.005 = 0.115 > 1.5 * 2 pi / 300 = 0.0314
    outside = owner.enumerate_owners(spec, context, 300.0, ('Earth',), S, body_n_range=(exact + 0.005, exact + 0.03), body_varpi_rate=0.0)
    assert '7E-23+0+16' not in keys(outside)


def test_rate_bound_uses_the_faster_of_the_two_periods():
    assert owner.rate_bound(2000.0, 4000.0, S) == pytest.approx(1.5 * TWO_PI / 2000.0)
    assert owner.rate_bound(float('nan'), 3000.0, S) == pytest.approx(1.5 * TWO_PI / 3000.0)
    assert math.isnan(owner.rate_bound(float('nan'), float('nan'), S))


def test_theory_force_by_hand():
    # 2 |k_o| n a mu_Mars = 2 * 20 * 1.84 * 2.27 * mu_M
    spec = ResonanceSpec.from_key('11M-20+0+9')
    assert owner.theory_force(spec, 1.84, 2.27) == pytest.approx(2 * 20 * 1.84 * 2.27 * const.PLANETS_MASS['Mars'])
    three = ResonanceSpec.from_key('1V+5J-7+0+0+1')
    mu = const.PLANETS_MASS['Venus'] * const.PLANETS_MASS['Jupiter']
    assert owner.theory_force(three, 1.84, 2.27) == pytest.approx(2 * 7 * 1.84 * 2.27 * mu)


def test_dominant_period_finds_a_sine_under_a_trend():
    t = np.arange(0.0, 40000.0, 2.0)
    rng = np.random.default_rng(1)
    values = 3e-4 * np.sin(TWO_PI * t / 1965.0) + 1e-8 * t + 2e-5 * rng.standard_normal(len(t))
    period = owner.dominant_period(t, values, 500.0, 10000.0)
    assert period == pytest.approx(1965.0, rel=0.01)
    assert math.isnan(owner.dominant_period(t, values, 10.0, 5.0))


# --- acceptance ----------------------------------------------------------------------------------------


def _candidate(r2=0.9, coherence=0.95, psi=0.02, strength=0.01):
    return OwnerCandidate(
        spec=ResonanceSpec.from_key('11M-20+0+9'),
        rate_rad_yr=0.0,
        r2_median=r2,
        r2_windows=(r2, r2),
        phi0_coherence=coherence,
        psi_share=psi,
        force_au_yr=strength,
        force_theory_au_yr=1.0,
        strength_ratio=strength,
        rank=1,
        accepted=False,
        reason='',
    )


def _accept(candidate, sigma_r2, **kwargs):
    return owner.accept(candidate, sigma_r2, S, **kwargs)


def test_accept_order_of_the_tests_and_the_three_routes():
    assert _accept(_candidate(r2=0.69), 0.0) == (False, owner.REASON_R2)
    assert _accept(_candidate(coherence=0.79), 0.0) == (False, owner.REASON_COHERENCE)
    assert _accept(_candidate(psi=0.11), 0.0) == (False, owner.REASON_PSI)
    # sigma turns back below the owner's band by more than the owner can move it: the owner moves a ripple on it
    assert _accept(_candidate(), 0.0, sigma_slow_turns=1) == (False, owner.REASON_RIPPLE)
    assert _accept(_candidate(), 0.0, sigma_slow_turns=0) == (True, 'accepted:r2')
    assert _accept(_candidate(strength=10.5), 0.0) == (True, 'accepted:r2')  # W3-F3: the mu scale is no bound
    # r2: 0.9 >= 0.6 + 0.2
    assert _accept(_candidate(), 0.6) == (True, 'accepted:r2')
    # sigma fits as well (the identity) and keeps its phase: not better
    assert _accept(_candidate(), 0.95, sigma_coherence=0.9) == (False, owner.REASON_NOT_BETTER)
    # phase: 0.95 >= 0.5 + 0.2
    assert _accept(_candidate(), 0.95, sigma_coherence=0.5) == (True, 'accepted:phase')
    # sigma's force over its mu scale is no route (W3-F3)
    assert _accept(_candidate(), 0.95, sigma_coherence=0.9, sigma_strength_ratio=50.0) == (False, owner.REASON_NOT_BETTER)


def test_slow_oscillation_by_hand():
    """sigma = slow sine (A_s, period 20 kyr) + band sine (A_b, 2 kyr) on [0, 60 kyr]: the slow part has 6 turning
    points (extrema at 5, 15, ..., 55 kyr), each of prominence 2 A_s; the band's peak-to-peak is 2 A_b."""
    t = np.arange(0.0, 60000.0, 10.0)
    region = np.ones_like(t, dtype=bool)
    slow, band = 1.0 * np.sin(TWO_PI * t / 20000.0), 0.05 * np.sin(TWO_PI * t / 2000.0)
    turns, ratio = owner.slow_oscillation(t, slow, band, region, 2.0)
    assert turns == 6
    # band peak-to-peak 0.1; slow 2 plus what the removed line adds: slope -12 / (w L^2) = -1.06e-5 /yr, at most
    # 2 x 0.318 over the half-spans of 30 kyr
    assert 2.0 / 0.1 <= ratio <= (2.0 + 2 * 0.318) / 0.1
    # a slow oscillation below 2 x the band's peak-to-peak is not counted: prominence 2 x 0.04 < 2 x 2 x 0.05
    turns, _ = owner.slow_oscillation(t, 0.04 * np.sin(TWO_PI * t / 20000.0), band, region, 2.0)
    assert turns == 0
    # a circulating slow sigma has no turning points at any amplitude
    assert owner.slow_oscillation(t, 1e-3 * t + 0.2 * np.sin(TWO_PI * t / 20000.0), band, region, 2.0)[0] == 0
    # only the turning points inside the region count: the first half holds 3 (5, 15, 25 kyr)
    assert owner.slow_oscillation(t, slow, band, t < 30000.0, 2.0)[0] == 3


def test_ripple_turns_count_in_forced_wave_regimes_only():
    """A librating owner's slow motion reaches sigma through the identity; a circulating one forces no slow part."""
    nan = float('nan')
    regime = owner.RegimeFit(owner.REGIME_LIBRATION, (0.0,), 0.9, (0.9,), 0.95, 1.0, 0.1, 0.2, 1.0, 2, 5.0)
    assert owner.ripple_turns(regime) == 0
    assert owner.ripple_turns(replace(regime, regime=owner.REGIME_FORCED_POSITIVE)) == 2
    assert owner.ripple_turns(replace(regime, regime=owner.REGIME_FORCED_NEGATIVE, sigma_slow_turns=0, sigma_slow_to_band=nan)) == 0


def test_owner_required():
    assert owner.owner_required('pass', 1.0, True, S)
    assert owner.owner_required('not_measured', 0.2, False, S)  # mandatory below 45 deg
    assert not owner.owner_required('not_applicable', float('nan'), False, S)
    assert not owner.owner_required('not_measured', 0.2, True, settings_from_config(mmr_owner_search='off'))


# --- synthetic records -------------------------------------------------------------------------------------

STEP, SPAN = 1.0, 30000.0
P_LIB, AMPLITUDE = 2000.0, 0.6
N_MARS, N_JUPITER = 3.3405, 0.52965
VARPI_RATE = TWO_PI / 40000.0
PSI_RATE = TWO_PI / 15000.0


def _planet(name, t, n, a, phase):
    zeros = np.zeros_like(t)
    return PlanetSeries(name, t, a + zeros, 0.05 + zeros, 0.02 + zeros, 1.0 + zeros, 2.0 + zeros, np.mod(n * t + phase, TWO_PI))


def synthetic(key: str) -> MMRSeries:
    """The body driven by 11M-20+0+9 (see the module docstring); `key` is the angle analysed."""
    t = np.arange(0.0, SPAN + STEP / 2, STEP)
    omega = TWO_PI / P_LIB
    owner_angle = math.pi + AMPLITUDE * np.sin(omega * t)
    varpi = 1.0 + VARPI_RATE * t
    lambda_mars = N_MARS * t + 0.3
    mean_longitude = (11 * lambda_mars + 9 * varpi - owner_angle) / 20
    n = (11 * N_MARS + 9 * VARPI_RATE - AMPLITUDE * omega * np.cos(omega * t)) / 20
    a = (TWO_PI / n) ** (2 / 3)
    n_mean = (11 * N_MARS + 9 * VARPI_RATE) / 20
    n_venus = 7 * n_mean - 5 * N_JUPITER - VARPI_RATE + PSI_RATE  # sigma's lambda part turns at PSI_RATE
    planets = {
        'Venus': _planet('Venus', t, n_venus, 0.723, 0.7),
        'Mars': _planet('Mars', t, N_MARS, 1.524, 0.3),
        'Jupiter': _planet('Jupiter', t, N_JUPITER, 5.203, 1.1),
    }
    spec = ResonanceSpec.from_key(key)
    angle = spec.body_coefficient * mean_longitude + spec.body_varpi_coefficient * varpi
    for name, k, p in zip(spec.planets, spec.planet_coefficients, spec.planet_varpi_coefficients):
        angle = angle + k * (planets[name].mean_longitude) + p * planets[name].varpi
    Omega = np.mod(1.4 - TWO_PI / 60000.0 * t, TWO_PI)
    series = MMRSeries(
        t_yr=t,
        a=a,
        e=0.2 + np.zeros_like(t),
        inc=0.07 + np.zeros_like(t),
        Omega=Omega,
        omega=np.mod(varpi - Omega, TWO_PI),
        varpi=varpi,
        angle_wrapped=np.mod(angle, TWO_PI),
        resonance=spec,
        regime=Regime(step_yr=STEP, span_yr=SPAN, n_rows=len(t), planets_saved=tuple(planets)),
        planets=planets,
        mean_longitude=np.mod(mean_longitude, TWO_PI),
        name='synthetic',
    )
    series.validate()
    return series


def _inputs(series):
    context = build_context(series, S)
    sampling = analyse_sampling(series, context, S)
    return preprocess(series, context, sampling, S), context


@pytest.fixture(scope='module')
def induced():
    series = synthetic('1V+5J-7+0+0+1')
    prep, context = _inputs(series)
    return series, prep, context, owner.run_owner(series, prep, context, P_LIB, S)


def test_synthetic_induced_angle_finds_its_owner(induced):
    series, _, _, result = induced
    assert result.check == CheckValue.FAIL
    assert result.accepted is not None and result.accepted.spec.key == '11M-20+0+9'
    assert result.detail == str(OriginDetail.OWNER_LIBRATION)  # A = 0.6 rad: the owner never turns
    assert result.freq_period_yr == pytest.approx(P_LIB, rel=0.02)
    best = result.accepted
    assert best.r2_median > 0.9 and best.phi0_coherence > 0.95
    assert best.psi_share < 0.05  # psi holds no lambda: sigma's wave is all (7/20) phi_o
    assert best.strength_ratio < 1  # the owner's force is reachable with mu_Mars
    assert result.sigma_r2 < best.r2_median - S.mmr_owner_r2_gain
    # psi turns at the constructed rate (one turn per 15 kyr)
    assert result.psi_rate_rad_yr == pytest.approx(PSI_RATE, rel=0.05)
    columns = result.to_columns()
    assert columns['own_name'] == '11M-20+0+9' and columns['own_detail'] == 'owner_libration'


def test_synthetic_owner_angle_has_no_owner():
    series = synthetic('11M-20+0+9')
    prep, context = _inputs(series)
    result = owner.run_owner(series, prep, context, P_LIB, S)
    assert result.check == CheckValue.PASS and result.accepted is None
    assert all(c.spec.key != '11M-20+0+9' for c in result.candidates)


def test_without_longitudes_or_for_a_retrograde_angle(induced):
    series, prep, context, _ = induced
    no_planets = replace(series, planets={})
    assert owner.run_owner(no_planets, prep, context, P_LIB, S).check == CheckValue.NOT_MEASURED
    no_lambda = replace(series, mean_longitude=None)
    result = owner.run_owner(no_lambda, prep, context, P_LIB, S)
    assert result.check == CheckValue.NOT_MEASURED and owner.FLAG_NO_LONGITUDES in result.flags
    retro = replace(series, resonance=ResonanceSpec.from_key('1J-3r0'))
    assert owner.run_owner(retro, prep, context, P_LIB, S).check == CheckValue.NOT_APPLICABLE
    off = owner.run_owner(series, prep, context, P_LIB, settings_from_config(mmr_owner_search='off'))
    assert off.check == CheckValue.NOT_MEASURED


def test_owner_angle_matches_the_constructed_one(induced):
    series, _, context, _ = induced
    spec = ResonanceSpec.from_key('11M-20+0+9')
    planet_n = {'Mars': owner.planet_mean_motion(context, 'Mars')}
    phi = owner.owner_angle(spec, series, planet_n)
    expected = math.pi + AMPLITUDE * np.sin(TWO_PI / P_LIB * series.t_yr)
    offset = np.round((phi[0] - expected[0]) / TWO_PI) * TWO_PI  # the unwrap starts at the wrapped value
    assert np.max(np.abs(phi - offset - expected)) < 1e-6


def test_windows_keep_clear_of_the_band_pass_edges(induced):
    series, prep, _, _ = induced
    frame = owner.build_frame(prep, series, P_LIB, S)
    trim = S.filter_edge_trim_cutoffs * S.mmr_owner_band_factor * P_LIB
    inside = frame.t_yr[frame.mask]
    assert inside[0] >= trim - STEP and inside[-1] <= SPAN - trim + STEP
