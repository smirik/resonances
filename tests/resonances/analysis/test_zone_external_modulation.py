"""The separatrix zone reads its strength modulation from motion external to the angle (near_separatrix fix of
2026-11-20, docs/physics/geometry.md "The separatrix zone"): e and the members' offset varpi - varpi_P are low-passed
at LIBRATION_CENTRE_CYCLES x the angle's own period P_own (`zone.external_cutoff_yr`); the members count when their fit
reconstructs a_dot out of sample (`zone.members_held_out_gain`), whatever beta sweeps (Codex review of 84af9cd, F3);
the trend of e is removed before its edge hold, and dK/K and the strength come from that one decomposition (F4); a
record too short to resolve the effective window max(cutoff, W_c), librators too (F5), has a standing separatrix,
zone [1, 1], flag zone_external_unresolved (never an absent zone).

Expected values by hand: the e^q half-range dK/K = q de / e_bar of the slow term alone (PHYSICS B1a), the Butterworth
order-2 forward-backward gain 1 / (1 + (P_c / P)^4) at the own line, the multiplet algebra W = C0 + C1 e^{i beta}.
"""

import dataclasses

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from resonances.resonance.analysis import filtering
from resonances.resonance.analysis.mmr import models as m
from resonances.resonance.analysis.mmr import zone
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.context import table_mean_motion
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import Flag
from tests.resonances.analysis.builders import make

S = AnalysisSettings()
KEY = '2J-1+0-1'  # p = -1, Jupiter's term 0: q = 1, the members j = 0, 1


def _series(t, e, varpi_planet=None, key=KEY):
    spec = m.ResonanceSpec.from_key(key)
    n = len(t)
    planets = {}
    if varpi_planet is not None:
        ones = np.ones(n)
        planets['Jupiter'] = m.PlanetSeries('Jupiter', t, 5.2 * ones, 0.048 * ones, 0.02 * ones, 1.75 * ones, varpi_planet)
    return m.MMRSeries(
        t_yr=t,
        a=np.full(n, 3.28),
        e=np.asarray(e, dtype=float),
        inc=np.zeros(n),
        Omega=np.zeros(n),
        omega=np.full(n, 1.0),
        varpi=np.full(n, 1.0),
        angle_wrapped=np.zeros(n),
        resonance=spec,
        regime=m.Regime(step_yr=float(t[1] - t[0]), span_yr=float(t[-1] - t[0]), n_rows=n),
        planets=planets,
    )


def _slow(t, e, window_yr):
    """The slow components as `preprocess.slow_components` makes them for a circulation: W_c = 3 P_c (here e itself,
    whose lines all pass a W_c of 30 yr)."""
    return make(m.SlowComponents, window_yr=window_yr, t_yr=t, e_slow=np.asarray(e, float), window=filtering.edge_window(t, window_yr))


def _context(c=-1.0):
    return make(
        m.Context,
        identity_coefficient=c,
        planet_mass={'Jupiter': 1e-3},
        planet_varpi_rate_rad_yr={},
        body_varpi_rate_rad_yr=float('nan'),
    )


# ------------------------------------------------------------------------------------------------ e^q route


def test_the_angles_own_e_line_is_not_a_strength_modulation():
    """e = 0.1 + 0.01 cos(2 pi t / P_own) + 0.02 cos(2 pi t / 20 kyr), P_own = 200 yr over 100 kyr.

    The old window W_c = 3 P_c = 30 yr keeps both lines: dK/K = (0.01 + 0.02) / 0.1 = 0.3. At 3 P_own = 600 yr the own
    line keeps a gain 1 / (1 + 3^4) = 1/82 and the secular one 1 / (1 + 0.03^4) = 1: dK/K = (0.02 + 0.01/82) / 0.1 = 0.201.
    """
    p_own, p_sec = 200.0, 20000.0
    t = np.arange(0.0, 100000.0, 5.0)
    e = 0.1 + 0.01 * np.cos(2 * np.pi * t / p_own) + 0.02 * np.cos(2 * np.pi * t / p_sec)
    series, slow, context = _series(t, e), _slow(t, e, 30.0), _context()
    old = zone.eccentricity_modulation(series, slow, context, t)
    assert old.eq_dk_over_k == pytest.approx(0.3, rel=0.02)
    assert zone.external_cutoff_yr(p_own, S) == pytest.approx(3 * p_own)
    new = zone.eccentricity_modulation(series, slow, context, t, p_own, S)
    assert new.eq_dk_over_k == pytest.approx(0.2 + 0.1 / 82, rel=0.02)
    assert not new.external_unresolved
    # the strength follows the secular term alone: Omega^2 / Omega_bar^2 = e_slow / e_bar
    inside = filtering.window_mask(t, filtering.edge_window(t, 2 * zone.external_cutoff_yr(p_own, S)))
    truth = (0.1 + 0.02 * np.cos(2 * np.pi * t / p_sec)) / 0.1
    assert np.max(np.abs(new.omega_squared[inside] - truth[inside])) < 0.01
    # within a cutoff of the edges the filter would spread the own line back (0.008 of its 0.01, a strength swing of
    # 0.16): e is held flat there (what is left is the fitted trend of e, 1e-6 here)
    edge = t < zone.external_cutoff_yr(p_own, S)
    assert np.ptp(new.omega_squared[edge]) < 1e-5


def test_a_librator_keeps_its_slow_window():
    """A W_c already at libration_centre_cycles x P_own (a librator: W_c = 3 P_cyc) is not filtered again."""
    t = np.arange(0.0, 100000.0, 5.0)
    e = 0.1 + 0.01 * np.cos(2 * np.pi * t / 200.0) + 0.02 * np.cos(2 * np.pi * t / 20000.0)
    series, slow, context = _series(t, e), _slow(t, e, 600.0), _context()
    kept = zone.eccentricity_modulation(series, slow, context, t, 200.0, S)
    assert kept.eq_dk_over_k == pytest.approx(zone.eccentricity_modulation(series, slow, context, t).eq_dk_over_k, rel=1e-12)


def test_a_record_too_short_for_the_cutoff_has_a_standing_separatrix():
    """1000 yr against 2 x 3 x 200 yr = 1200 yr: no stretch of the record is a cutoff away from both edges. e (a 1500-yr
    wave of 20 %) is then its mean: Omega^2 constant, dK/K = 0, flag zone_external_unresolved."""
    t = np.arange(0.0, 1000.0, 1.0)
    e = 0.1 + 0.02 * np.cos(2 * np.pi * t / 1500.0)
    series, slow, context = _series(t, e), _slow(t, e, 30.0), _context()
    assert zone.eccentricity_modulation(series, slow, context, t).eq_dk_over_k > 0.05  # the old reading
    new = zone.eccentricity_modulation(series, slow, context, t, 200.0, S)
    assert new.external_unresolved
    assert new.eq_dk_over_k == 0.0
    assert np.allclose(new.omega_rel, 1.0)
    # 2 x 3 x 160 yr = 960 yr < 1000 yr: resolved again
    assert not zone.eccentricity_modulation(series, slow, context, t, 160.0, S).external_unresolved


@pytest.mark.parametrize('slow_window', [30.0, 600.0, 601.0])
def test_the_resolution_is_checked_on_every_branch(slow_window):
    """F5: the same 1000-yr record with e low-passed at W_c = 30 yr (a circulation: refiltered at the 600-yr cutoff),
    W_c = 600 yr (a librator: W_c = 3 P_cyc is the cutoff itself) and 601 yr (longer): the effective window
    max(600, W_c) leaves no sample 600 yr from both edges, so the strength is constant and flagged on every branch
    (84af9cd checked only the refiltered one: a strength swing of 0.12 from an edge-dominated filter, no flag)."""
    t = np.arange(0.0, 1000.0, 1.0)
    e = 0.1 + 0.02 * np.cos(2 * np.pi * t / 1500.0)
    slow = _slow(t, filtering.lowpass_with(t, e, slow_window, S), slow_window)
    found = zone.eccentricity_modulation(_series(t, e), slow, _context(), t, 200.0, S)
    assert found.external_unresolved
    assert np.ptp(found.omega_squared) < 1e-12 and abs(found.eq_dk_over_k) < 1e-12


def test_an_edge_window_of_one_sample_is_unresolved():
    """Span exactly 2 x 600 yr on a 1-yr grid: the edge-free window [600, 600] holds one sample, which resolves nothing
    (a trend and a half-range need two); a span of 1202 yr leaves three samples and is resolved. q = 0 (no power of e
    in the term): nothing to resolve, never flagged."""
    for span, unresolved in ((1200.0, True), (1202.0, False)):
        t = np.arange(0.0, span + 0.5, 1.0)
        e = 0.1 + 0.02 * np.cos(2 * np.pi * t / 1500.0)
        found = zone.eccentricity_modulation(_series(t, e), _slow(t, e, 600.0), _context(), t, 200.0, S)
        assert found.external_unresolved == unresolved
    t = np.arange(0.0, 1000.0, 1.0)
    e = 0.1 + 0.02 * np.cos(2 * np.pi * t / 1500.0)
    flat = zone.eccentricity_modulation(_series(t, e, key='3J-3+0+0'), _slow(t, e, 600.0), _context(), t, 200.0, S)
    assert flat.source == zone.SOURCE_NONE and not flat.external_unresolved


def test_a_pure_trend_of_e_is_no_modulation_after_the_edge_hold():
    """F4: e = 0.1 + 0.1 t / 60 kyr, own period 4000 yr: the cutoff 12 kyr refilters e (W_c 60 yr) and holds it within
    12 kyr of the edges. Removing the trend after holding e turned the line into a clipped ramp (84af9cd: a strength
    swing of 0.128, zone [0.975, 1.05] from no oscillation); removed first, e_osc is a constant: no strength
    modulation anywhere, dK/K = 0, the trend 0.1 / 60 kyr = 1.667e-3 / kyr, and the zone is the standing [1, 1]."""
    t = np.arange(0.0, 60000.0, 5.0)
    e = 0.1 + 0.1 * t / 60000.0
    found = zone.eccentricity_modulation(_series(t, e), _slow(t, filtering.lowpass_with(t, e, 60.0, S), 60.0), _context(), t, 4000.0, S)
    assert not found.external_unresolved
    assert np.ptp(found.omega_squared) < 1e-9
    assert abs(found.eq_dk_over_k) < 1e-9
    assert found.e_trend_per_kyr == pytest.approx(0.1 / 60.0, rel=1e-6)
    standing, _ = zone._zone_of(found, np.ones_like(t, dtype=bool), 2 * np.pi / 4000.0, float('nan'), S)
    assert (standing.j_lo, standing.h_hi) == pytest.approx((1.0, 1.0))


# ------------------------------------------------------------------------------------------------ the members


def _members_record(turns, span=60000.0, step=5.0, c=-0.75):
    """sigma_ddot = c a_dot, a_dot = Im[W e^{i sigma}], W = C0 + C1 e^{i beta}, beta = varpi - varpi_P = -varpi_P + 1
    sweeping `turns` turns over the record (SEPARATRIX §2.3); Jupiter's varpi is saved."""
    c0, c1 = 2e-7 * np.exp(0.4j), 1e-7 * np.exp(1.1j)
    nu = 2 * np.pi * turns / span
    t = np.arange(0.0, span, step)

    def amplitude(x):
        return c0 + c1 * np.exp(1j * nu * x)

    sol = solve_ivp(
        lambda x, y: [y[1], c * np.imag(amplitude(x) * np.exp(1j * y[0]))], (t[0], t[-1]), [2.0, 0.0], t_eval=t, rtol=1e-10, atol=1e-13
    )
    sigma = sol.y[0]
    adot = np.imag(amplitude(t) * np.exp(1j * sigma))
    varpi_planet = 1.0 - nu * t  # beta = varpi - varpi_P = nu t
    return t, sigma, adot, varpi_planet


OVERFIT_KEY = '36J-13+0-23'  # p = -23, Jupiter's term 0: the members j = 0..23 (24 harmonics, 17886 36M-61 has 26)


def _overfit_record(span=20000.0, step=5.0, c=-0.75):
    """A single harmonic C0 sin(sigma + arg C0) (small period 500 yr) plus a force it does not explain: Gaussian noise
    (seed 1) low-passed at 100 yr, of the harmonic's amplitude. beta = varpi - varpi_P makes one turn over the record,
    so the 24 members of OVERFIT_KEY fit that noise in sample with 48 coefficients on 6.7 blocks of the 1.5-kyr cutoff
    (17886 36M-61: gain 0.76 in sample, -2495 out of sample)."""
    scale = (2 * np.pi / 500.0) ** 2 / abs(c)
    c0 = scale * np.exp(0.4j)
    t = np.arange(0.0, span, step)
    sol = solve_ivp(lambda x, y: [y[1], c * np.imag(c0 * np.exp(1j * y[0]))], (t[0], t[-1]), [0.0, 0.0], t_eval=t, rtol=1e-10, atol=1e-13)
    sigma = sol.y[0]
    noise = filtering.lowpass_with(t, np.random.default_rng(1).standard_normal(len(t)), 100.0, S)
    adot = np.imag(c0 * np.exp(1j * sigma)) + noise * abs(c0) / np.std(noise)
    return t, sigma, adot, 1.0 - 2 * np.pi * t / span


def _librating_record(span=120000.0, step=5.0, c=-0.75):
    """beta = 1.2 sin(2 pi t / 20 kyr) librates (range 2.4 rad, six cycles), W = C0 + C1 e^{i beta}, |C1| / |C0| = 0.8,
    scaled to a small period of 500 yr (Codex review of 84af9cd, F3): the frozen small periods 373-486 yr are far below
    the 20-kyr forcing."""
    scale = (2 * np.pi / 500.0) ** 2 / abs(c)
    c0, c1 = scale * np.exp(0.4j), 0.8 * scale * np.exp(1.1j)
    t = np.arange(0.0, span, step)

    def amplitude(x):
        return c0 + c1 * np.exp(1j * 1.2 * np.sin(2 * np.pi * x / 20000.0))

    sol = solve_ivp(
        lambda x, y: [y[1], c * np.imag(amplitude(x) * np.exp(1j * y[0]))],
        (t[0], t[-1]),
        [-np.angle(amplitude(0.0)) + 0.25, 0.0],
        t_eval=t,
        rtol=1e-10,
        atol=1e-13,
        method='DOP853',
    )
    sigma = sol.y[0]
    adot = np.imag(amplitude(t) * np.exp(1j * sigma))
    return t, sigma, adot, 1.0 - 1.2 * np.sin(2 * np.pi * t / 20000.0), abs(c) * np.abs(amplitude(t))


@pytest.mark.parametrize('turns', [0.5, 3.0])
def test_members_count_when_their_fit_carries_out_of_sample(turns):
    """F3: the members' harmonics are told apart by how beta moves, not by how far: on half a turn as on three the
    exact multiplet's fit recovers |C1| / |C0| = 0.5 and predicts the blocks of the 1.5-kyr cutoff it was not fitted
    on as well as in sample. P_own = 500 yr: the 60-kyr record resolves 3 x 500 yr."""
    t, sigma, adot, varpi_planet = _members_record(turns)
    series = _series(t, np.full_like(t, 0.1), varpi_planet)
    mask = np.ones_like(t, dtype=bool)
    context = _context(-0.75)
    found = zone.best_members(series, t, sigma, adot, mask, context, 20.0, S, 500.0)
    assert found.eps == pytest.approx((0.5,), rel=0.02)  # |C1| / |C0| by hand
    assert found.gain >= S.mmr_force_multi_gain and found.held_out_gain == pytest.approx(found.gain, abs=0.01)
    zone_mod, _ = zone.modulations(series, _slow(t, series.e, 60.0), context, t, sigma, adot, mask, 20.0, 500.0, S)
    assert zone_mod.source == zone.SOURCE_MEMBERS and not zone_mod.members_unresolved
    # too short a record for 3 x P_own: no members (the e^q route then has a standing separatrix)
    assert zone.best_members(series, t, sigma, adot, mask, context, 20.0, S, 20000.0) is None


def test_a_librating_secular_offset_modulates_the_zone():
    """F3, Codex's counterexample to the one-turn rule: beta librates over 2.4 rad, six cycles in 120 kyr. 84af9cd
    dropped the members (range < 2 pi) and the zone read a constant strength; the fit is identified (eps 0.8 by hand,
    out of sample too) and the zone's strength follows |c W(t)| (relative swing 0.48)."""
    t, sigma, adot, varpi_planet, strength = _librating_record()
    series = _series(t, np.full_like(t, 0.1), varpi_planet)
    mask = np.ones_like(t, dtype=bool)
    context = _context(-0.75)
    zone_mod, _ = zone.modulations(series, _slow(t, series.e, 60.0), context, t, sigma, adot, mask, 20.0, 500.0, S)
    assert zone_mod.source == zone.SOURCE_MEMBERS
    assert zone_mod.eps == pytest.approx((0.8,), rel=0.01)
    assert zone_mod.held_out_gain >= S.mmr_force_multi_gain
    inside = filtering.window_mask(t, filtering.edge_window(t, 1500.0))  # beyond the low-pass edge of the offset
    assert np.max(np.abs(zone_mod.omega_squared[inside] / strength[inside] - 1)) < 0.01
    assert np.ptp(zone_mod.omega_rel**2) == pytest.approx(0.483, abs=0.01)


def test_a_fit_that_does_not_carry_out_of_sample_is_an_unresolved_modulation():
    """24 members on one turn of beta fit 0.15 of a_dot's variance in sample (above mmr_force_multi_gain 0.1) and
    predict the blocks they were not fitted on far worse than the single harmonic (held-out gain < -1): the members'
    modulation is not identified. The zone reads e^q and says why (`members_unresolved`, flag
    zone_members_unresolved): not an absent modulation."""
    t, sigma, adot, varpi_planet = _overfit_record()
    series = _series(t, np.full_like(t, 0.1), varpi_planet, key=OVERFIT_KEY)
    mask = np.ones_like(t, dtype=bool)
    context = _context(-0.75)
    found = zone.best_members(series, t, sigma, adot, mask, context, 20.0, S, 500.0)
    assert found.gain >= S.mmr_force_multi_gain  # taken in sample
    assert found.held_out_gain < -1.0
    zone_mod, _ = zone.modulations(series, _slow(t, series.e, 60.0), context, t, sigma, adot, mask, 20.0, 500.0, S)
    assert zone_mod.source == zone.SOURCE_ECCENTRICITY
    assert zone_mod.members_unresolved and zone_mod.held_out_gain == found.held_out_gain
    assert str(Flag.ZONE_MEMBERS_UNRESOLVED) in zone._zone_flags(zone_mod, None, [], False)


def test_held_out_blocks_interpolate_and_do_not_extrapolate_beta():
    """The blocks are of the offset's cutoff: a nearly fixed beta (nu T = 0.006 rad) whose basis follows a smooth, real
    strength variation W = C0 (1 + 0.8 sin(2 pi t / T)) predicts the blocks between its fitted ones (held-out gain
    about the in-sample 0.13): the zone may follow that variation, it is the force's own. An exact multiplet on half a
    turn, the case time halves would have to extrapolate, carries too (`test_members_count_when_...`)."""
    span, step, c = 60000.0, 5.0, -0.75
    c0 = 2e-7 * np.exp(0.4j)
    t = np.arange(0.0, span, step)

    def amplitude(x):
        return c0 * (1 + 0.8 * np.sin(2 * np.pi * x / span))

    sol = solve_ivp(
        lambda x, y: [y[1], c * np.imag(amplitude(x) * np.exp(1j * y[0]))], (t[0], t[-1]), [2.0, 0.0], t_eval=t, rtol=1e-10, atol=1e-13
    )
    sigma = sol.y[0]
    adot = np.imag(amplitude(t) * np.exp(1j * sigma))
    series = _series(t, np.full_like(t, 0.1), 1.0 - 0.006 * t / span)
    found = zone.best_members(series, t, sigma, adot, np.ones_like(t, dtype=bool), _context(c), 20.0, S, 500.0)
    assert found.gain >= S.mmr_force_multi_gain
    assert found.held_out_gain == pytest.approx(found.gain, abs=0.01)


def test_the_members_own_fit_stays_a_frame_when_the_zone_drops_it():
    """`zone.modulations`: identification is the zone's only. On the unidentified 24 members the zone reads e^q
    (constant e here: m = 1), while the frames keep the members' own fit at P_c, the same object `best_members` gives without a
    period (the reading in which R0 passed on UG22 10E-17, 117306 11M-20); on three turns both read the members (the
    zone's at the external cutoff), and without a period both are the old members' fit."""
    context = _context(-0.75)
    for (t, sigma, adot, varpi_planet), key, zone_source in (
        (_overfit_record(), OVERFIT_KEY, zone.SOURCE_ECCENTRICITY),
        (_members_record(3.0), KEY, zone.SOURCE_MEMBERS),
    ):
        series = _series(t, np.full_like(t, 0.1), varpi_planet, key=key)
        slow = _slow(t, series.e, 60.0)
        mask = np.ones_like(t, dtype=bool)
        own = zone.best_members(series, t, sigma, adot, mask, context, 20.0, S)
        zone_mod, frame_mod = zone.modulations(series, slow, context, t, sigma, adot, mask, 20.0, 500.0, S)
        assert zone_mod.source == zone_source
        assert frame_mod.source == zone.SOURCE_MEMBERS
        assert frame_mod.gain == own.gain and frame_mod.eps == own.eps  # the old fit, not the low-passed one
        assert np.array_equal(frame_mod.omega_squared, own.omega_squared)
        # without the angle's period there is no external cutoff: one reading for both
        same_zone, same_frame = zone.modulations(series, slow, context, t, sigma, adot, mask, 20.0, float('nan'), S)
        assert same_zone is same_frame and same_zone.gain == own.gain


# ------------------------------------------------------------------------------------------------ through the analysis

P0_YR = 400.0
W0 = 2 * np.pi / P0_YR


def _rotation_series(h, span_yr, e_of_t, key='3J-1'):
    """A free rotation sigma_ddot = -w0^2 sin sigma of energy h (3J-1 body whose lambda and a follow sigma exactly,
    Jupiter on its mean line) with the body's e given as a function of time."""
    t = np.arange(0.0, span_yr + 0.5, 1.0)
    sol = solve_ivp(
        lambda _, y: [y[1], -(W0**2) * np.sin(y[0])], (t[0], t[-1]), [0.0, 2 * W0 * np.sqrt(h)], t_eval=t, rtol=1e-10, atol=1e-12
    )
    sigma, sigma_dot = sol.y
    spec = m.ResonanceSpec.from_key(key)
    ones = np.ones_like(t)
    n_j = table_mean_motion('Jupiter')
    lam_j = n_j * t + 0.3
    k_j, k, p, varpi = spec.planet_coefficients[0], spec.body_coefficient, spec.body_varpi_coefficient, 1.0
    lam = (sigma - k_j * lam_j - p * varpi) / k
    n = (sigma_dot - k_j * n_j) / k
    jupiter = m.PlanetSeries('Jupiter', t, 5.2026 * ones, 0.048 * ones, 0.0228 * ones, 1.753 * ones, 0.257 * ones, np.mod(lam_j, 2 * np.pi))
    return m.MMRSeries(
        t_yr=t,
        a=(2 * np.pi / n) ** (2.0 / 3.0),
        e=e_of_t(t),
        inc=0.05 * ones,
        Omega=0.4 * ones,
        omega=(varpi - 0.4) * ones,
        varpi=varpi * ones,
        angle_wrapped=np.mod(k_j * lam_j + k * lam + p * varpi, 2 * np.pi),
        resonance=spec,
        regime=m.Regime(float(t[1] - t[0]), float(t[-1] - t[0]), len(t)),
        planets={'Jupiter': jupiter},
        mean_longitude=np.mod(lam, 2 * np.pi),
        name='synthetic',
    )


def test_an_unresolved_record_keeps_a_zone_of_one():
    """A rotation at h = 1.1 (turn 2 K(1/h) / (w0 sqrt h) = 318 yr by hand) over 1500 yr, with e varying by 20 % over
    1000 yr: 2 x 3 x 318 yr > 1500 yr, so nothing slower than the angle is resolved. The zone is the standing
    separatrix [1, 1] (J_lo = h_hi = 1), flagged, and not absent."""
    from scipy.special import ellipk

    h = 1.1
    turn = 2 * ellipk(1 / h) / (W0 * np.sqrt(h))
    assert turn == pytest.approx(318.4, abs=0.5)
    assert 2 * 3 * turn > 1500.0
    analysis = analyse_mmr(
        _rotation_series(h, 1500.0, lambda t: 0.15 + 0.03 * np.sin(2 * np.pi * t / 1000.0)), S, scan_subresonances=False, scan_owner=False
    )
    energy = analysis.energy
    assert energy is not None and energy.zone is not None
    assert energy.zone.j_lo == pytest.approx(1.0) and energy.zone.h_hi == pytest.approx(1.0)
    assert str(Flag.ZONE_EXTERNAL_UNRESOLVED) in energy.flags


def test_a_short_librator_has_a_standing_separatrix():
    """F5 through the analysis (Codex's short_librator): a libration at h = 0.15 in sigma_ddot = -w0^2 m(t) sin sigma,
    m = 1 + 0.2 sin(2 pi t / 3000 yr), e = 0.15 sqrt(m) (q = 2 for 3J-1), over 1500 yr. P_cyc = 385 yr: the effective
    window 3 P_cyc = 1155 yr leaves no sample a window from both edges (2 x 1155 > 1500), so the zone is the flagged
    standing [1, 1] (84af9cd: [0.975, 1.025] from an edge-dominated filter, no flag); the status stays resonant."""
    from tests.resonances.independent_w18.test_w18_separatrix import _series as w18_series

    t = np.arange(0.0, 1501.0, 1.0)

    def strength(x):
        return 1 + 0.2 * np.sin(2 * np.pi * x / 3000.0)

    sol = solve_ivp(
        lambda x, y: [y[1], -(W0**2) * strength(x) * np.sin(y[0])],
        (t[0], t[-1]),
        [0.0, 2 * W0 * np.sqrt(0.15)],
        t_eval=t,
        rtol=2e-11,
        atol=1e-13,
        method='DOP853',
    )
    series = dataclasses.replace(w18_series(t, sol.y[0], sol.y[1]), e=0.15 * np.sqrt(strength(t)))
    analysis = analyse_mmr(series, S, scan_subresonances=False, scan_owner=False)
    assert analysis.geometry.period_cycle_yr == pytest.approx(385.0, abs=5.0)
    assert 2 * 3 * analysis.geometry.period_cycle_yr > t[-1] - t[0]
    energy = analysis.energy
    assert (energy.zone.j_lo, energy.zone.h_hi) == pytest.approx((1.0, 1.0))
    assert str(Flag.ZONE_EXTERNAL_UNRESOLVED) in energy.flags
    assert str(analysis.verdict.status) == 'resonant'
