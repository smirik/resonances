"""W18 moving centre: libration about the slowly moving minimum of the averaged potential (geometry: the band test in
psi = sigma - c(t), `mmr.centre_track`) and R0's period in the full averaged well.

Independent truths:
- the Codex witness of W17 Q3: turning points 0, 100, 60, 160, 120 deg pass the bands by +10 deg and fail the
  half-swing rule by -20 deg (by hand below);
- a linear drift r t of the cycle centres at t = 0..N-1: about the best fixed centre RMS = r sqrt((N^2 - 1) / 12), about
  the time-reversed track r sqrt((N^2 - 1) / 3) (by hand);
- the Plutino well of a circular Neptune lies at 180 deg (test_impl_w9_strength_well, M&D sect. 8); the angle
  2N-3+1+0 = (2 lambda_N - 3 lambda + varpi) + varpi_N - varpi with varpi_N = 0 then has its well at 180 deg - varpi(t):
  a body whose varpi turns 60 deg over the record carries the well -60 deg, a track known without the code;
- the pendulum period 4 K(sin^2(A/2)) / omega0 (M&D 1999 eq. 9.? / any mechanics text); the period of the Plutino well
  from an own averaging (explicit Kepler solve, a plain loop over the nodes) integrated with scipy's solve_ivp;
- W17 numbers of the benchmark records (Q3 fix/diag_runs.csv: psi margins 27011 4M-7 +122.4 deg, 65366 6M-11 +158.1 deg;
  Q4: 2003 UG22 cosine period law 0.77, full well 0.98-1.10 cycle by cycle; Q8: 2013 AS27 track RMS 1.4 deg against
  14.9 and 29 deg). The record tests are skipped without the benchmark data.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.special import ellipk

from resonances.resonance.analysis import geometry as geo
from resonances.resonance.analysis.mmr import averaged_potential as ap
from resonances.resonance.analysis.mmr import centre_track, io
from resonances.resonance.analysis.mmr import strength as st
from resonances.resonance.analysis.mmr.coorbital import Orbit
from resonances.resonance.analysis.mmr.models import MMRSeries, PlanetSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleSubtype, Status

S = settings_from_config()
RULE = st.quadrature(S)
DEG = np.pi / 180
NEPTUNE_AXIS = 30.07
NEPTUNE_MASS = 5.1503e-5
PLUTINO_AXIS = NEPTUNE_AXIS / (2 / 3) ** (2 / 3)
CIRCLE = Orbit(0.0, 0.0, 0.0, 0.0)


# --- margins and the validity of a track, by hand ---------------------------------------------------------------------


def test_band_and_half_swing_margins_of_the_codex_witness():
    values = np.radians([0.0, 100.0, 60.0, 160.0, 120.0])
    _, mids, amps = geo.cycle_centres(np.arange(5.0), values)
    # half-cycle midpoints 50, 80, 110, 140 and half-amplitudes 50, 20, 50, 20: cycles 65, 95, 125 +- 35
    assert np.degrees(mids) == pytest.approx([65.0, 95.0, 125.0])
    assert np.degrees(amps) == pytest.approx([35.0, 35.0, 35.0])
    assert np.degrees(geo.band_margin(mids, amps)) == pytest.approx(10.0)  # (65 + 35) - (125 - 35)
    assert np.degrees(geo.half_swing_margin(values)) == pytest.approx(-20.0)  # lowest maximum 100, highest minimum 120


def test_track_rms_and_its_two_nulls_by_hand():
    n, rate = 10, 0.02
    times = np.arange(float(n))
    mids = rate * times
    exact = geo.MovingWell(geo.WELL_OK, times, rate * times)
    rms, fixed, reversed_ = geo.well_track_rms(times, mids, exact)
    assert rms == pytest.approx(0.0, abs=1e-15)
    assert fixed == pytest.approx(rate * np.sqrt((n**2 - 1) / 12))
    assert reversed_ == pytest.approx(rate * np.sqrt((n**2 - 1) / 3))
    assert geo.well_track_valid(rms, fixed, reversed_)
    # the same drift the wrong way: it is the reversed track that fits, and the track loses to it
    backwards = geo.MovingWell(geo.WELL_OK, times, -rate * times)
    assert not geo.well_track_valid(*geo.well_track_rms(times, mids, backwards))
    # a well that does not move can never beat the best fixed centre
    still = geo.MovingWell(geo.WELL_OK, times, np.full(n, float(np.mean(mids))))
    assert not geo.well_track_valid(*geo.well_track_rms(times, mids, still))


def test_bands_hold_the_bottom_of_the_well():
    # cycle bands m +- A in psi: every one must hold psi = 0, the predicted centre
    assert geo.bands_hold_bottom(np.radians([2.0, -3.0, 1.0]), np.radians([20.0, 22.0, 25.0]))
    assert not geo.bands_hold_bottom(np.radians([2.0, 30.0, 1.0]), np.radians([20.0, 22.0, 25.0]))  # one beside it
    assert not geo.bands_hold_bottom(np.zeros(0), np.zeros(0))


# --- the psi frame in the geometry -----------------------------------------------------------------------------------

AMPLITUDE = 20 * DEG
PERIOD = 1000.0


def _turning_points(centre, t):
    sigma = centre(t) + AMPLITUDE * np.sin(2 * np.pi * t / PERIOD)
    return geo.find_turning_points(t, sigma, 1 * DEG, (t[0], t[-1]))


class CountingModel:
    """A geometry.WellModel returning a fixed track (nodes `t`, `centre`) and counting its calls."""

    def __init__(self, t, centre):
        self.well = geo.MovingWell(geo.WELL_OK, np.asarray(t, float), np.asarray(centre, float))
        self.calls = 0

    def __call__(self, times, values):
        self.calls += 1
        return self.well


def test_adiabatic_drift_of_three_amplitudes_passes_in_psi_and_fails_in_sigma():
    t = np.arange(0.0, 12000.0, 5.0)
    drift = -3 * AMPLITUDE / t[-1]  # the centre moves 3 A (60 deg) over 12 cycles: 5 deg per cycle against A = 20 deg

    def centre(x):
        return np.pi + drift * x

    tp = _turning_points(centre, t)
    record = (t[0], t[-1])
    without = geo._find_episodes(tp, record, record, S)
    assert not any(e.accepted for e in without)
    assert {str(e.reject_reason) for e in without} <= {'centre_moves', 'edge'}

    model = CountingModel(t, centre(t))
    episodes = geo._find_episodes(tp, record, record, S, well_model=model)
    accepted = [e for e in episodes if e.accepted]
    assert len(accepted) == 1 and model.calls >= 1
    episode = accepted[0]
    assert episode.shape == geo.EpisodeShape.MOVING_CENTRE and episode.frame == geo.FRAME_PSI
    assert episode.well_status == geo.WELL_OK and episode.centre_stationarity == 'pass'
    assert episode.margin_sigma_rad < 0 < episode.margin_psi_rad
    # in psi the oscillation is the bare sine: every band is the same, the margin 2 A (less the TP offset of
    # (drift P / (2 pi A))^2 / 2 ~ 1e-3 of A, the sigma extrema lying off the psi extrema)
    assert np.degrees(episode.margin_psi_rad) == pytest.approx(2 * 20.0, abs=0.5)
    assert np.degrees(episode.well_drift_rad) == pytest.approx(np.degrees(drift * (episode.t_end_yr - episode.t_start_yr)), abs=0.1)
    assert episode.well_rms_rad < 1e-3 < episode.well_rms_fixed_rad < episode.well_rms_reversed_rad


def test_a_wrong_well_never_vetoes_a_sigma_pass_and_never_accepts_a_drift():
    t = np.arange(0.0, 12000.0, 5.0)
    record = (t[0], t[-1])
    # a stationary libration: the sigma test passes and the model is never asked (16017 7S-1: a track running -1750 deg)
    wild = CountingModel(t, np.pi - 30.0 * t / t[-1])
    stationary = [e for e in geo._find_episodes(_turning_points(lambda x: np.full_like(x, np.pi), t), record, record, S, 0, wild)]
    assert [e.shape for e in stationary if e.accepted] == [geo.EpisodeShape.STATIONARY]
    assert wild.calls == 0 and stationary[0].frame == geo.FRAME_SIGMA and stationary[0].well_status == ''
    # a drifting centre with the time-reversed well: the track fails its nulls, nothing is accepted in psi
    drift = -3 * AMPLITUDE / t[-1]
    reversed_well = CountingModel(t, np.pi + drift * (t[-1] - t))
    episodes = geo._find_episodes(_turning_points(lambda x: np.pi + drift * x, t), record, record, S, 0, reversed_well)
    assert reversed_well.calls >= 1 and not any(e.accepted for e in episodes)
    assert geo.WELL_INVALID in {e.well_status for e in episodes}


# --- the track of a real well: the Plutino well carried by varpi -----------------------------------------------------


def _plutino_record(varpi_turn_rad, key='2N-3+1+0', span=20000.0, step=10.0):
    """A Plutino (a, e = 0.1, i = 0) whose varpi turns `varpi_turn_rad` over the record, a saved circular Neptune
    (varpi_N = 0). The angle values are filled by the caller."""
    t = np.arange(0.0, span, step)
    n = len(t)
    zeros, ones = np.zeros(n), np.ones(n)
    varpi = varpi_turn_rad * t / t[-1]
    neptune = PlanetSeries('Neptune', t, NEPTUNE_AXIS * ones, zeros, zeros, zeros, zeros)
    spec = ResonanceSpec.from_key(key)
    return MMRSeries(
        t, PLUTINO_AXIS * ones, 0.1 * ones, zeros, zeros, varpi, varpi, zeros, spec, Regime(step, t[-1], n), {'Neptune': neptune}
    )


def test_track_of_the_plutino_well_follows_180_minus_varpi():
    series = _plutino_record(60 * DEG)
    context = SimpleNamespace(planet_mass={'Neptune': NEPTUNE_MASS})
    well = centre_track.build(series, context, S, 100.0)
    assert well is not None and well.node_step == pytest.approx(max(100.0, (series.t_yr[-1] - series.t_yr[0]) / S.mmr_centre_track_nodes))
    t = series.t_yr
    times = np.array([2000.0, 2500.0, 3000.0, 17000.0])
    track = well(times, np.radians([180.0, 200.0, 160.0, 135.0]))
    assert track.status == centre_track.WELL_OK
    expected = np.pi - 60 * DEG * track.t_yr / t[-1]  # the varpi of the node, averaged over P_c about it (linear: exact)
    assert np.degrees(track.centre_rad) == pytest.approx(np.degrees(expected), abs=0.5)
    # the geometry with this model: a sine about the moving well is accepted in psi, not in sigma
    tp = _turning_points(lambda x: np.pi - 60 * DEG * x / t[-1], t)
    record = (t[0], t[-1])
    accepted = [e for e in geo._find_episodes(tp, record, record, S, 0, well) if e.accepted]
    assert len(accepted) == 1 and accepted[0].shape == geo.EpisodeShape.MOVING_CENTRE
    assert np.degrees(accepted[0].well_drift_rad) == pytest.approx(-60 * (accepted[0].t_end_yr - accepted[0].t_start_yr) / t[-1], abs=1.0)


def test_no_track_without_the_planet_or_for_three_body():
    series = _plutino_record(0.0)
    context = SimpleNamespace(planet_mass={'Neptune': NEPTUNE_MASS})
    assert centre_track.build(replace(series, planets={}), context, S, 100.0) is None
    assert centre_track.availability(replace(series, planets={}), context) == centre_track.PLANET_NOT_SAVED
    assert centre_track.availability(series, SimpleNamespace(planet_mass={})) == centre_track.NO_PLANET_MASS
    three = replace(series, resonance=ResonanceSpec.from_key('4J-2S-1'))
    assert centre_track.availability(three, context) == centre_track.THREE_BODY


# --- R0: the period in the full well ---------------------------------------------------------------------------------


@pytest.mark.parametrize('amplitude', [0.1, 0.5, 1.5, 2.5, 3.0])
def test_period_in_profile_is_the_pendulum_period(amplitude):
    omega0 = 2 * np.pi / 150.0
    sigma = np.linspace(0.0, 2 * np.pi, S.mmr_strength_sigma_grid, endpoint=False)
    turns = np.concatenate([sigma - 2 * np.pi, sigma, sigma + 2 * np.pi])
    potential = np.tile(omega0**2 * (1 - np.cos(sigma - 1.0)), 3)
    period, far = ap.period_in_profile(turns, potential, 1.0 + amplitude)
    assert period == pytest.approx(4 * ellipk(np.sin(amplitude / 2) ** 2) / omega0, rel=1e-5)
    assert far == pytest.approx(1.0 - amplitude, abs=1e-4)


@pytest.mark.parametrize('start', [1.02, 0.97, 1.004])
def test_period_in_profile_finds_a_swing_narrower_than_the_grid(start):
    # Codex W18: a swing between two nodes of the 5-deg grid (V = 1 - cos(sigma - 1), omega0 = 1: T -> 2 pi), and the
    # start itself a root of V = E where V falls (never taken for the far turning point)
    sigma = np.linspace(0.0, 2 * np.pi, S.mmr_strength_sigma_grid, endpoint=False)
    turns = np.concatenate([sigma - 2 * np.pi, sigma, sigma + 2 * np.pi])
    period, far = ap.period_in_profile(turns, np.tile(1 - np.cos(sigma - 1.0), 3), start)
    amplitude = abs(start - 1.0)
    assert period == pytest.approx(4 * ellipk(np.sin(amplitude / 2) ** 2), rel=1e-3)  # the spline's error near the bottom
    assert far == pytest.approx(1.0 - (start - 1.0), abs=1e-3)


@pytest.mark.parametrize('start, speed', [(1.5, 0.4), (0.2, 0.9), (1.0, 0.5), (-2.0, 0.05)])
def test_orbit_in_profile_with_a_speed_is_the_pendulum_of_its_energy(start, speed):
    # V = 1 - cos(sigma - 1), omega0 = 1: the orbit through sigma0 with speed u has 1 - cos A = 1 - cos(sigma0 - 1) + u^2 / 2,
    # its period 4 K(sin^2(A / 2)) and its turning points 1 -+ A (by hand)
    sigma = np.linspace(0.0, 2 * np.pi, S.mmr_strength_sigma_grid, endpoint=False)
    turns = np.concatenate([sigma - 2 * np.pi, sigma, sigma + 2 * np.pi])
    amplitude = np.arccos(np.cos(start - 1.0) - speed**2 / 2)
    period, lower, upper = ap.orbit_in_profile(turns, np.tile(1 - np.cos(sigma - 1.0), 3), start, speed**2 / 2)
    assert period == pytest.approx(4 * ellipk(np.sin(amplitude / 2) ** 2), rel=1e-4)
    assert (lower, upper) == pytest.approx((1.0 - amplitude, 1.0 + amplitude), abs=1e-4)
    # above the top (1 - cos A > 2): no turning point, a turn
    assert ap.orbit_in_profile(turns, np.tile(1 - np.cos(sigma - 1.0), 3), start, 2.0 + speed)[0] == np.inf


@pytest.mark.parametrize('start', [0.8, 1.5, -0.8, 0.2])
def test_orbit_in_profile_passes_a_lower_top_on_either_side(start):
    # Codex W18 round 2: V = -cos x + 0.5 cos 2x has two minima at +-1.32 and a lower top V(0) = -0.5 inside the outer
    # tops V(pi) = 1.5. The orbit of E = 0 is one orbit over the inner top, whatever point it is started from: its
    # bounds and period from scipy's brentq and quad on the analytic V (independent of the spline)
    x = np.linspace(-np.pi, np.pi, S.mmr_strength_sigma_grid, endpoint=False)
    turns = np.concatenate([x - 2 * np.pi, x, x + 2 * np.pi])

    def potential(value):
        return -np.cos(value) + 0.5 * np.cos(2 * value)

    bound = brentq(potential, 1.0, 3.0)
    period = 2 * quad(lambda value: 1 / np.sqrt(-2 * potential(value)), -bound, bound, limit=400)[0]
    result = ap.orbit_in_profile(turns, np.tile(potential(x), 3), start, -potential(start))
    assert result == pytest.approx((period, -bound, bound), rel=1e-5)


class _TranslatingCosineWell:
    """The `WellAlongRecord.periods` of V = w^2 (1 - cos(sigma - v t)) on the package's quadrature grid (the analytic
    profile stands for <D>; the well's motion is known without the code)."""

    available = True

    def __init__(self, omega0, rate):
        self.omega0, self.rate = omega0, rate

    def periods(self, time_yr, starts_rad, kinetic=None):
        centre = self.rate * time_yr

        def line(nodes):
            return SimpleNamespace(
                nodes_per_turn=nodes,
                sigma=lambda m: np.asarray(m) * 2 * np.pi / nodes,
                evaluate=lambda m: (
                    self.omega0**2 * (1 - np.cos(np.asarray(m) * 2 * np.pi / nodes - centre)),
                    self.omega0**2 * (1 - np.cos(np.asarray(m) * 2 * np.pi / nodes - centre)),
                    np.full(len(m), 100.0),
                ),
            )

        return ap.periods_from_turning_points(line, starts_rad, 1.0, RULE, kinetic).periods


@pytest.mark.parametrize('amplitude, advance', [(1.0, 0.0), (1.0, 0.3), (3.1, 0.0), (3.1, 0.08), (3.1, 0.3)])
def test_full_well_period_of_a_libration_in_a_translating_well(amplitude, advance):
    """Codex W18 (integration, finding 1): sigma = v t + q, q the pendulum q'' = -w^2 sin q from rest at A (scipy), an
    exact libration about a well translating by `advance` rad per period. At a TP of sigma q_dot = -v, not 0: started at
    rest there the 3.1-rad orbit gave T_obs / T_pred = 0.77 at 0.08 rad per period. Its true period is 4 K(sin^2(A/2)) / w."""
    omega0 = 0.01
    period = 4 * ellipk(np.sin(amplitude / 2) ** 2) / omega0
    rate = advance / period
    t = np.arange(0.0, 6 * period, 0.5)
    q = solve_ivp(lambda _, y: [y[1], -(omega0**2) * np.sin(y[0])], (0.0, t[-1]), [amplitude, 0.0], t_eval=t, rtol=1e-11, atol=1e-12).y[0]
    sigma = rate * t + q
    turning = geo.find_turning_points(t, sigma, 1e-3, (t[0], t[-1]))
    times, values = turning.times_yr, turning.values_rad
    bounds = [(times[i], times[i + 2], False) for i in range(len(times) - 2)]
    predicted = centre_track.full_well_periods(_TranslatingCosineWell(omega0, rate), bounds, times, values)
    law = [(end - start) / p for (start, end, _), p in zip(bounds, predicted)]
    assert np.all(np.isfinite(law))
    assert np.median(law) == pytest.approx(1.0, abs=2e-3)


def _own_plutino_well(sigma_grid, e=0.1, nodes=720):
    """<D>(sigma) of the Plutino (sigma = 2 lambda_N - 3 lambda + varpi, varpi = 0) and a circular coplanar Neptune,
    averaged over 3 turns of lambda_N by a plain loop (own Kepler solve), independent of `LineAverage`."""
    lam_n = 2 * np.pi * np.arange(3 * nodes) / nodes
    out = []
    for s in sigma_grid:
        mean_anomaly = (2 * lam_n - s) / 3
        big_e = mean_anomaly.copy()
        for _ in range(50):
            big_e = mean_anomaly + e * np.sin(big_e)
        x, y = PLUTINO_AXIS * (np.cos(big_e) - e), PLUTINO_AXIS * np.sqrt(1 - e * e) * np.sin(big_e)
        xn, yn = NEPTUNE_AXIS * np.cos(lam_n), NEPTUNE_AXIS * np.sin(lam_n)
        direct = 1 / np.hypot(x - xn, y - yn)
        indirect = (x * xn + y * yn) / NEPTUNE_AXIS**3
        out.append(np.mean(direct - indirect))
    return np.asarray(out)


def test_full_well_period_of_the_plutino_against_an_own_averaging_and_ode():
    k = 3
    n = 2 * np.pi / PLUTINO_AXIS**1.5
    scale = 3 * k**2 * NEPTUNE_MASS * n**2 * PLUTINO_AXIS
    grid = np.linspace(0.0, 2 * np.pi, 361)[:-1]
    values = scale * _own_plutino_well(grid)
    force = CubicSpline(np.append(grid, 2 * np.pi), np.append(values, values[0]), bc_type='periodic').derivative()

    def own_period(start):
        def rhs(_, y):
            return [y[1], -force(np.mod(y[0], 2 * np.pi))]

        def turn(_, y):
            return y[1]

        solution = solve_ivp(rhs, [0, 1e5], [start, 0.0], events=turn, rtol=1e-10, atol=1e-12, max_step=200.0)
        times = solution.t_events[0]
        return float(times[times > 1.0][1])

    body = ap.Ellipse(PLUTINO_AXIS, Orbit(0.1, 0.0, 0.0, 0.0))
    planet = ap.Ellipse(NEPTUNE_AXIS, CIRCLE)
    starts = np.radians([150.0, 120.0, 250.0])

    def line(nodes):
        return ap.LineAverage(body, planet, -3, 2, 0.0, nodes)

    result = ap.periods_from_turning_points(line, starts, scale, RULE)
    assert result.reason == ap.REASON_NONE and result.closest_au > 0
    assert result.in_well  # the swing 120-250 deg holds the minimum at 180 deg and not the top at 0 deg
    for start, period in zip(starts, result.periods):
        assert period == pytest.approx(own_period(start), rel=2e-3)
    # the well of a Plutino is symmetric about 180 deg: 150 and 210 deg are one orbit
    mirror = ap.periods_from_turning_points(line, [np.radians(210.0)], scale, RULE)
    assert mirror.periods[0] == pytest.approx(result.periods[0], rel=1e-4)
    # a swing about 0 deg, the top of the Plutino well, is no orbit of it: the caller reads no period from it
    assert not ap.periods_from_turning_points(line, np.radians([-40.0, 40.0]), scale, RULE).in_well


# --- benchmark records ------------------------------------------------------------------------------------------------


def _analysed(example):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    analysis = analyse_mmr(io.load_series(example), S)
    return analysis, analysis.numbers_row()


@pytest.mark.benchmark
def test_ug22_20kyr_is_resonant_by_the_period_of_its_full_well():
    analysis, numbers = _analysed('mmr2__2003_UG22__10E-17+0+7')
    # the fitted pendulum misses the flat-bottomed well (Q4: 0.77), the full well holds it (Q4: 0.98-1.10 per cycle)
    assert numbers['sep_period_law_fitted_med'] == pytest.approx(0.77, abs=0.02)
    assert numbers['sep_period_law_source'] == 'full_well' and numbers['sep_period_law_med'] == pytest.approx(1.0, abs=0.08)
    assert numbers['sep_R0_sigma_only'] == 'pass'
    assert analysis.verdict.status == Status.RESONANT


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'example, psi_margin_deg',
    [('mmr2__27011__4M-7+0+3', 122.4), ('mmr2__65366__6M-11+0+5', 158.1)],
)
def test_mars_resonances_librate_about_the_moving_minimum(example, psi_margin_deg):
    analysis, numbers = _analysed(example)
    episode = next(e for e in analysis.geometry.runs if e.accepted and e.frame == geo.FRAME_PSI)
    assert episode.shape == geo.EpisodeShape.MOVING_CENTRE and episode.well_status == geo.WELL_OK
    assert episode.margin_sigma_rad < 0 and np.degrees(episode.margin_psi_rad) == pytest.approx(psi_margin_deg, abs=2.0)
    assert episode.well_rms_rad < episode.well_rms_fixed_rad / 10  # 2-3 deg against 56-81 deg
    # the full well predicts the period cycle by cycle (the fitted pendulum: 0.79-0.91)
    assert numbers['sep_period_law_med'] == pytest.approx(1.0, abs=0.05)


@pytest.mark.benchmark
def test_277617_quasi_satellite_capture_stays_transient():
    analysis, _ = _analysed('coorbital__277617__1M-1+0+0__20kyr')
    assert analysis.verdict.status == Status.TRANSIENT
    assert analysis.verdict.angle_subtype != AngleSubtype.MOVING_CENTRE


@pytest.mark.benchmark
def test_2013_as27_31_36_librates_about_the_moving_minimum_of_earth():
    """2013 AS27 31E-36+0+5 is no benchmark example of its own: the 31:36 angle with Earth is built from the series of
    mmr3__2013_AS27__4E+9S-5 and its run's planets (W17 Q8). Q8 (own tools): the centre follows the minimum of Earth's
    well -56 deg in 20 kyr, RMS 1.4 deg against 14.9 (best fixed centre) and 29 deg (reversed track); W17 Q3: band
    margins -8.2 deg in sigma, +35.1 deg in psi; the period 2-8 % shorter than the well's."""
    example = 'mmr3__2013_AS27__4E+9S-5+0+0-8'
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    series = io.load_series(example)
    if 'Earth' not in series.planets:
        pytest.skip("the run's planets are not available")
    earth = series.planets['Earth']
    angle = 31 * np.asarray(earth.mean_longitude) - 36 * np.asarray(series.mean_longitude) + 5 * np.asarray(series.varpi)
    spec = ResonanceSpec.from_key('31E-36+0+5', nominal_axis_au=(36 / 31) ** (2 / 3))
    analysis = analyse_mmr(replace(series, resonance=spec, angle_wrapped=np.mod(angle, 2 * np.pi), angle_unwrapped_numpy=None), S)
    verdict, numbers = analysis.verdict, analysis.numbers_row()
    assert (verdict.status, verdict.angle_subtype) == (Status.RESONANT, AngleSubtype.MOVING_CENTRE)
    assert numbers['ep_frame'] == 'psi' and numbers['ep_n_tp'] >= 250  # one run of the whole record
    assert numbers['ep_margin_sigma_deg'] == pytest.approx(-8.2, abs=1.5) and numbers['ep_margin_psi_deg'] == pytest.approx(35.1, abs=1.5)
    assert numbers['ep_centre_track_rms_deg'] == pytest.approx(1.4, abs=0.5)
    assert numbers['ep_centre_fixed_rms_deg'] == pytest.approx(14.9, abs=1.5)
    assert numbers['ep_centre_reversed_rms_deg'] == pytest.approx(29.0, abs=2.0)
    assert numbers['ep_centre_model_drift_deg'] == pytest.approx(-56.0, abs=3.0)
    assert 0.92 - 0.02 <= numbers['sep_period_law_med'] <= 0.98 + 0.02 and numbers['sep_R0_sigma_only'] == 'pass'
