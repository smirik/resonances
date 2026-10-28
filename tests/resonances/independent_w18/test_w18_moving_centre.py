"""Independent tests of W18 item 1 (docs/physics/geometry.md "Runs and episodes", the moving centre; docs/physics/response.md
"Energy and potential", the period in the full averaged well): the moving-centre acceptance, a wrong track that must not help, and R0's period in the
full well.

The reference here is our own averaged disturbing function, written from the definition and checked against Murray &
Dermott (1999), Table 8.5 (f_d of 3:1 = 0.5988, f_27 of 2:1 = -1.1905): D = 1/|r - r_P| - r.r_P/r_P^3 (direct +
indirect, per unit G m_P, 3D Keplerian orbits), averaged over lambda_P at fixed sigma by brute force. The records are
integrated in that well: sigma_ddot = -3 k^2 mu_J (GM / a^2) d<D>/dsigma (Lagrange's equations; the doc's
V = 3 k^2 mu_P n^2 a <D>). The well moves because the body's omega drifts: for 3J-1 at i = 20 deg the term
s^2 cos(sigma + 2 omega) dominates e^2 cos sigma, so the minimum moves by about -2.3 x the change of omega.

So a record built in a moving well IS a libration about a moving centre (Gallardo 2006, §3), with the period of the
full well, and the package must find both; a record about a fixed centre must pass in sigma whatever the track says;
a record that follows the track backwards in time must not be rescued by it.
"""

import functools

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq

from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.averaged_potential import orbit_in_profile
from resonances.resonance.analysis.mmr.context import table_mean_motion
from resonances.resonance.analysis.mmr.models import MMRSeries, PlanetSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.settings import AnalysisSettings

SETTINGS = AnalysisSettings()
MU_JUPITER = 1.0 / 1047.3486
GM_SUN = 4 * np.pi**2  # au^3 / yr^2, Gaussian years
JUPITER = (5.2026, 0.048, 0.0228, 1.753, 0.257 - 1.753)  # a, e, i, Omega, omega
A_BODY, E_BODY, I_BODY, NODE_BODY = 2.5, 0.05, np.radians(20.0), 0.4
SPAN_YR = 30000.0
OMEGA_START, OMEGA_END = np.radians(100.0), np.radians(190.0)
SIGMA_PAD = np.radians(30.0)


# --------------------------------------------------------------------------------------------------------------------
# our own averaged disturbing function


def _kepler(mean_anomaly, e):
    ecc = mean_anomaly + e * np.sin(mean_anomaly)
    for _ in range(30):
        ecc = ecc - (ecc - e * np.sin(ecc) - mean_anomaly) / (1 - e * np.cos(ecc))
    return ecc


def _position(a, e, inc, node, peri, mean_anomaly):
    ecc = _kepler(np.mod(mean_anomaly, 2 * np.pi), e)
    x, y = a * (np.cos(ecc) - e), a * np.sqrt(1 - e * e) * np.sin(ecc)
    cn, sn, cp, sp, ci, si = np.cos(node), np.sin(node), np.cos(peri), np.sin(peri), np.cos(inc), np.sin(inc)
    return np.stack(
        [
            (cn * cp - sn * sp * ci) * x + (-cn * sp - sn * cp * ci) * y,
            (sn * cp + cn * sp * ci) * x + (-sn * sp + cn * cp * ci) * y,
            sp * si * x + cp * si * y,
        ]
    )


def averaged_d(sigma, body, planet, k_p, k, p, n_nodes=1024):
    """<D>(sigma) [1/au], sigma = k_p lambda_P + k lambda + p varpi; lambda_P = theta runs over |k| turns."""
    a, e, inc, node, peri = body
    varpi = node + peri
    theta = 2 * np.pi * np.arange(n_nodes * abs(k)) / n_nodes
    r_p = _position(*planet, theta - planet[3] - planet[4])
    r_p3 = np.sum(r_p**2, axis=0) ** 1.5
    out = []
    for s in np.atleast_1d(sigma):
        lam = (s - k_p * theta - p * varpi) / k
        r = _position(a, e, inc, node, peri, lam - varpi)
        out.append(np.mean(1.0 / np.sqrt(np.sum((r - r_p) ** 2, axis=0)) - np.sum(r * r_p, axis=0) / r_p3))
    return np.array(out)


def test_reference_well_reproduces_the_classical_coefficients():
    """Planar, circular Jupiter, small e: (<D>(0) - <D>(pi)) / 2 = f e^q / a_J (M&D Table 8.5)."""
    a_j = 5.2026
    for key, (k_p, k, p, alpha, e, f) in {
        '3J-1': (3, -1, -2, (1 / 3) ** (2 / 3), 0.01, 0.5988),
        '2J-1': (2, -1, -1, 0.5 ** (2 / 3), 0.005, -1.1905),
    }.items():
        d = averaged_d(np.array([0.0, np.pi]), (alpha * a_j, e, 0.0, 0.4, 0.6), (a_j, 0.0, 0.0, 0.0, 0.0), k_p, k, p, n_nodes=4096)
        assert (d[0] - d[1]) / 2 * a_j / e ** abs(k_p + k) == pytest.approx(f, rel=2e-3), key


@functools.lru_cache(maxsize=None)
def _well_table():
    """<D>(omega, sigma) on a grid (sigma padded by 30 deg on both sides for a periodic spline)."""
    sigma = np.radians(np.arange(-30.0, 390.0, 2.5))
    omega = np.linspace(OMEGA_START - np.radians(10), OMEGA_END + np.radians(10), 41)
    table = np.array([averaged_d(sigma, (A_BODY, E_BODY, I_BODY, NODE_BODY, w), JUPITER, 3, -1, -2) for w in omega])
    return sigma, RectBivariateSpline(omega, sigma, table)


def _d(omega, sigma, order):
    _, spline = _well_table()
    return spline(omega, np.mod(sigma + SIGMA_PAD, 2 * np.pi) - SIGMA_PAD, dy=order, grid=False)


def _minimum(omega, start=None):
    """The well's minimum at omega by Newton from the grid's lowest point (or from `start`, on its branch)."""
    grid, _ = _well_table()
    s = grid[np.argmin(_d(omega, grid, 0))] if start is None else start
    for _ in range(50):
        s = s - _d(omega, s, 1) / _d(omega, s, 2)
    return float(s)


def _omega(t, reverse=False):
    x = 1 - t / SPAN_YR if reverse else t / SPAN_YR
    return OMEGA_START + (OMEGA_END - OMEGA_START) * x


def _track(times):
    """Our c(t): the minimum followed continuously along omega(t)."""
    track = [_minimum(_omega(times[0]))]
    for x in times[1:]:
        track.append(_minimum(_omega(x), track[-1]))
    return np.array(track)


# --------------------------------------------------------------------------------------------------------------------
# records in the well


SCALE = 3 * MU_JUPITER * GM_SUN / A_BODY**2  # 3 k^2 mu_J n^2 a with k = -1


def _series(t, sigma, sigma_dot, omega):
    spec = ResonanceSpec.from_key('3J-1')
    ones = np.ones_like(t)
    n_j = table_mean_motion('Jupiter')
    lam_j = n_j * t + 0.3
    varpi = NODE_BODY + omega
    varpi_dot = np.gradient(varpi, t)
    lam = (sigma - 3 * lam_j + 2 * varpi) / -1
    n = (sigma_dot - 3 * n_j + 2 * varpi_dot) / -1
    a_j, e_j, i_j, node_j, peri_j = JUPITER
    jupiter = PlanetSeries(
        'Jupiter', t, a_j * ones, e_j * ones, i_j * ones, node_j * ones, (node_j + peri_j) * ones, np.mod(lam_j, 2 * np.pi)
    )
    return MMRSeries(
        t_yr=t,
        a=(2 * np.pi / n) ** (2.0 / 3.0),
        e=E_BODY * ones,
        inc=I_BODY * ones,
        Omega=NODE_BODY * ones,
        omega=np.mod(omega, 2 * np.pi),
        varpi=varpi,
        angle_wrapped=np.mod(3 * lam_j - lam - 2 * varpi, 2 * np.pi),
        resonance=spec,
        regime=Regime(1.0, SPAN_YR, len(t)),
        planets={'Jupiter': jupiter},
        mean_longitude=np.mod(lam, 2 * np.pi),
        name='synthetic',
    )


@functools.lru_cache(maxsize=None)
def _record(dynamics, elements, swing_deg=55.0):
    """Integrate sigma in the well of omega_dyn(t) and save the elements of omega_el(t). `dynamics`, `elements`:
    'forward' (omega drifts 100 -> 190 deg), 'reverse' (190 -> 100 deg) or 'fixed' (100 deg)."""
    laws = {'forward': lambda x: _omega(x), 'reverse': lambda x: _omega(x, reverse=True), 'fixed': lambda x: OMEGA_START + 0 * x}
    omega_dyn, omega_el = laws[dynamics], laws[elements]
    t = np.arange(0.0, SPAN_YR + 0.5, 1.0)
    start = _minimum(omega_dyn(0.0)) + np.radians(swing_deg)
    sol = solve_ivp(
        lambda x, y: [y[1], -SCALE * _d(omega_dyn(x), y[0], 1)],
        (0.0, SPAN_YR),
        [start, 0.0],
        t_eval=t,
        rtol=1e-10,
        atol=1e-12,
        method='DOP853',
    )
    series = _series(t, sol.y[0], sol.y[1], omega_el(t))
    return analyse_mmr(series, SETTINGS, scan_subresonances=False, scan_owner=False)


def _accepted(analysis):
    rows = [row for row in analysis.episode_rows() if row['accepted']]
    assert len(rows) == 1
    return rows[0]


# --------------------------------------------------------------------------------------------------------------------
# acceptance on an adiabatic drift


def test_reference_track_moves_far_more_than_two_half_swings():
    """Our minimum moves by -110 to -170 deg while omega turns by 90 deg (the e^2 and e e_J terms pull against the s^2
    term); per kyr (about one libration) it moves < 10 deg against a 55-deg swing: adiabatic."""
    times = np.arange(0.0, SPAN_YR + 1, 1000.0)
    track = _track(times)
    drift = np.degrees(track[-1] - track[0])
    assert -170.0 < drift < -110.0
    assert np.max(np.abs(np.degrees(np.diff(track)))) < 10.0
    assert abs(drift) > 2 * 55.0


def test_libration_in_the_moving_well_is_accepted_in_psi():
    analysis = _record('forward', 'forward')
    verdict = analysis.verdict
    assert (str(verdict.status), str(verdict.angle_status), str(verdict.angle_subtype)) == ('resonant', 'libration', 'moving_centre')
    assert str(verdict.libration_origin) == 'dynamic'
    assert str(verdict.checks['check_mmr_a_response']) == 'pass'  # the force law, read in psi
    row = _accepted(analysis)
    assert row['frame'] == 'psi'
    assert row['centre_track'] == 'ok'
    assert row['margin_sigma_deg'] < 0 < row['margin_psi_deg']
    # the package's track against ours over the episode
    ours = _track(np.linspace(row['t_start_yr'], row['t_end_yr'], 150))  # dense: the minimum is followed, not jumped
    assert row['centre_model_drift_deg'] == pytest.approx(np.degrees(ours[-1] - ours[0]), abs=3.0)
    assert row['centre_track_rms_deg'] < row['centre_fixed_rms_deg']
    assert row['centre_track_rms_deg'] < row['centre_reversed_rms_deg']


def test_fixed_centre_passes_in_sigma_whatever_the_track():
    """The body librates in the well of omega = 100 deg while its saved omega drifts to 190 deg: the track moves by
    ~138 deg, the observed centre stands. The sigma test passes and the track is never consulted."""
    analysis = _record('fixed', 'forward')
    verdict = analysis.verdict
    assert (str(verdict.angle_status), str(verdict.angle_subtype)) == ('libration', 'stationary')
    row = _accepted(analysis)
    assert row['frame'] == 'sigma'
    assert row['margin_sigma_deg'] > 0
    assert abs(row['centre_shift_deg']) < 1.0


def test_a_track_run_backwards_does_not_rescue_a_drift():
    """The body follows the well of omega running 190 -> 100 deg, the saved omega runs 100 -> 190 deg: the observed
    centre moves +138 deg, the track -138 deg. It is the track reversed in time, so the track is no explanation and the
    run stays rejected in sigma."""
    analysis = _record('reverse', 'forward')
    verdict = analysis.verdict
    assert str(verdict.angle_subtype) != 'moving_centre'
    assert str(verdict.status) != 'resonant'
    rows = analysis.episode_rows()
    assert not any(row['accepted'] for row in rows)
    longest = max(rows, key=lambda row: row['n_tp'])
    assert longest['reject_reason'] == 'centre_moves'
    assert longest['frame'] != 'psi'
    assert longest['centre_reversed_rms_deg'] < longest['centre_track_rms_deg']


# --------------------------------------------------------------------------------------------------------------------
# R0: the period in the full well


def test_r0_period_law_is_one_in_the_moving_well():
    """The cycles were integrated in this very well: T_obs / T_pred = 1 if the package's full-well period, started at
    each TP with the well's speed, is right."""
    row = _record('forward', 'forward').numbers_row()
    assert row['sep_period_law_source'] == 'full_well'
    assert row['sep_period_law_n_full_well'] >= 20
    assert row['sep_period_law_med'] == pytest.approx(1.0, abs=0.01)


def _quadrature_period(omega, start, far):
    """2 int dsigma / sqrt(2 (E - V)) between the turning points, E = V(start), sigma = mid + half sin(theta) (no end
    singularity)."""
    energy = SCALE * _d(omega, start, 0)
    lower, upper = min(start, far), max(start, far)
    mid, half = (lower + upper) / 2, (upper - lower) / 2
    theta = np.linspace(-np.pi / 2, np.pi / 2, 200001)[1:-1]
    sigma = mid + half * np.sin(theta)
    kinetic = np.maximum(2 * (energy - SCALE * _d(omega, sigma, 0)), 1e-300)
    return 2 * np.trapezoid(half * np.cos(theta) / np.sqrt(kinetic), theta)


def _far_turning_point(omega, start, centre):
    """The root of V = V(start) on the other side of the centre (brentq on our spline)."""
    excess = lambda x: _d(omega, x, 0) - _d(omega, start, 0)  # noqa: E731
    side = np.sign(centre - start)
    grid = centre + side * np.radians(np.linspace(0.0, 179.0, 1791))
    first = int(np.argmax(excess(grid) >= 0))
    return brentq(excess, grid[first - 1], grid[first], xtol=1e-12)


@pytest.mark.parametrize('swing_deg', [130.0, 160.0])
def test_r0_period_law_is_one_in_an_anharmonic_fixed_well(swing_deg):
    """omega fixed at 100 deg, swings of 130 and 160 deg: our quadrature period agrees with the integrated cycle, and the
    package's full-well law with 1, where a cosine pendulum fitted to the swing would not."""
    analysis = _record('fixed', 'fixed', swing_deg)
    centre = _minimum(OMEGA_START)
    start = centre + np.radians(swing_deg)
    period = _quadrature_period(OMEGA_START, start, _far_turning_point(OMEGA_START, start, centre))
    row = _accepted(analysis)
    assert row['period_med_yr'] == pytest.approx(period, abs=2.0)  # sample times: within a step or two
    numbers = analysis.numbers_row()
    assert numbers['sep_period_law_source'] == 'full_well'
    assert numbers['sep_period_law_med'] == pytest.approx(1.0, abs=0.005)
    assert str(analysis.verdict.checks['check_mmr_potential']) == 'pass'


# --------------------------------------------------------------------------------------------------------------------
# orbit_in_profile against an ODE


W2 = (2 * np.pi / 500.0) ** 2


def _v(x):
    """Two wells and asymmetric tops: min near -31 deg, a shallow side well near 64 deg."""
    return W2 * (-np.cos(x) + 0.3 * np.cos(2 * x) + 0.15 * np.sin(3 * x))


def _dv(x):
    return W2 * (np.sin(x) - 0.6 * np.sin(2 * x) + 0.45 * np.cos(3 * x))


def _ode_orbit(start, kinetic):
    """(period, lower, upper) from sigma_ddot = -V'(sigma): events at sigma_dot = 0; at rest it starts downhill."""
    rate = np.sqrt(2 * kinetic) if kinetic > 0 else -1e-12 * np.sign(_dv(start))

    def turn(_, y):
        return y[1]

    sol = solve_ivp(lambda _, y: [y[1], -_dv(y[0])], (0.0, 20000.0), [start, rate], events=turn, rtol=1e-12, atol=1e-14, method='DOP853')
    times = sol.t_events[0][sol.t_events[0] > 1e-6]
    values = sol.y_events[0][sol.t_events[0] > 1e-6][:, 0]
    return times[2] - times[0], min(values[:2]), max(values[:2])


@pytest.mark.parametrize(
    'start_deg, kinetic',
    [(60.0, 0.0), (-120.0, 0.0), (20.0, 0.3 * W2), (-40.0, 0.8 * W2)],
    ids=['side-well-at-rest', 'main-well-at-rest', 'moving-start', 'over-the-side-top'],
)
def test_orbit_in_profile_matches_the_ode(start_deg, kinetic):
    grid = np.linspace(-np.pi, np.pi, 145)
    period, lower, upper = orbit_in_profile(grid, _v(grid), np.radians(start_deg), kinetic)
    period_ode, lower_ode, upper_ode = _ode_orbit(np.radians(start_deg), kinetic)
    assert period == pytest.approx(period_ode, rel=1e-4)
    assert (lower, upper) == pytest.approx((lower_ode, upper_ode), abs=1e-4)


def test_orbit_above_every_top_has_no_period():
    grid = np.linspace(-np.pi, np.pi, 145)
    xs = np.linspace(-np.pi, np.pi, 20001)
    above = np.max(_v(xs)) - _v(0.0) + 0.1 * W2
    period, lower, upper = orbit_in_profile(grid, _v(grid), 0.0, above)
    assert not np.isfinite(period) or not np.isfinite(lower) or not np.isfinite(upper)
