"""The separatrix zone reads its strength modulation from motion external to the angle (near_separatrix fix of
2026-11-20, docs/physics/geometry.md "The separatrix zone"): e and the members' offset varpi - varpi_P are low-passed
at LIBRATION_CENTRE_CYCLES x the angle's own period P_own (`zone.external_cutoff_yr`); the members count only when that
slow offset sweeps a full turn; a record too short to resolve the cutoff has a standing separatrix, zone [1, 1], flag
zone_external_unresolved (never an absent zone).

Expected values by hand: the e^q half-range dK/K = q de / e_bar of the slow term alone (PHYSICS B1a), the Butterworth
order-2 forward-backward gain 1 / (1 + (P_c / P)^4) at the own line, the multiplet algebra W = C0 + C1 e^{i beta}.
"""

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


@pytest.mark.parametrize('turns, counted', [(0.5, False), (3.0, True)])
def test_members_count_only_when_the_offset_sweeps_a_full_turn(turns, counted):
    """The members' harmonics j = 0, 1 are told apart only by the turning of beta; on half a turn the fit trades C1
    against C0 (523899 3N-5: nu T = 0.006 rad, eps 1-500). P_own = 500 yr: the 60-kyr record resolves 3 x 500 yr."""
    t, sigma, adot, varpi_planet = _members_record(turns)
    series = _series(t, np.full_like(t, 0.1), varpi_planet)
    mask = np.ones_like(t, dtype=bool)
    context = _context(-0.75)
    old = zone.best_members(series, t, sigma, adot, mask, context, 20.0, S)
    assert old is not None  # the old reading takes the members on any offset
    new = zone.best_members(series, t, sigma, adot, mask, context, 20.0, S, 500.0)
    assert (new is not None) == counted
    if counted:
        assert new.eps == pytest.approx((0.5,), rel=0.02)  # |C1| / |C0| by hand
    # too short a record for 3 x P_own: no members (the e^q route then has a standing separatrix)
    assert zone.best_members(series, t, sigma, adot, mask, context, 20.0, S, 20000.0) is None


def test_the_members_own_fit_stays_a_frame_when_the_zone_drops_it():
    """`zone.modulations`: the one-turn rule is the zone's only. On half a turn of beta the zone reads e^q (constant e
    here: m = 1), while the frames keep the members' own fit at P_c, the same object `best_members` gives without a
    period (the reading in which R0 passed on UG22 10E-17, 117306 11M-20); on three turns both read the members (the
    zone's at the external cutoff), and without a period both are the old members' fit."""
    context = _context(-0.75)
    for turns, zone_source in ((0.5, zone.SOURCE_ECCENTRICITY), (3.0, zone.SOURCE_MEMBERS)):
        t, sigma, adot, varpi_planet = _members_record(turns)
        series = _series(t, np.full_like(t, 0.1), varpi_planet)
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
