"""Independent functional tests of the W9 packages (strength well rule, decide G1l / rule C / L4, slow edge of an
episode, catalogue planets). Written from the W9 sections of impl/API.md, docs/libration-analysis.md and the public
docstrings only; every expected value is derived here (by hand or by an independent brute-force computation).

Synthetic MMR records are true pendulums in the averaged disturbing function of a circular, coplanar Neptune:
<D>(sigma) is averaged here by brute force over the planet's longitude at fixed sigma (direct + indirect part,
Murray & Dermott eq. 6.44-6.45), and sigma_ddot = -(3 k^2 / a^2) G m_P d<D>/dsigma is integrated with DOP853.
The body's a follows the identity sigma_dot = c (a - a*), c = -1.5 k n / a.
"""

import dataclasses
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from resonances.data import const
from resonances.resonance.analysis.geometry import TurningPoints, classify_run, find_episodes
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import averaged_potential as ap
from resonances.resonance.analysis.mmr import strength as strength_module
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.context import CATALOGUE_PLANETS, catalogue_longitudes
from resonances.resonance.analysis.mmr.coorbital import Orbit
from resonances.resonance.analysis.mmr.decide import AngleDecision, member_status, members_locked
from resonances.resonance.analysis.mmr.models import (
    GreatInequality,
    MMRSeries,
    Regime,
    ResonanceSpec,
    SubresonanceCandidate,
    SubresonanceResult,
)
from resonances.resonance.analysis.mmr.owner import SINC_HALF_SLOPE_MAX, catalogue_phase_error
from resonances.resonance.analysis.mmr.preprocess import predicted_ripple
from resonances.resonance.analysis.mmr.response import a_response_check, force_windows
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import (
    AngleStatus,
    AngleSubtype,
    CheckValue,
    LibrationOrigin,
    Reason,
    ResonanceType,
    Status,
    Verdict,
)

TWO_PI = 2 * np.pi


@pytest.fixture(scope='module')
def settings():
    return AnalysisSettings()


# ------------------------------------------------------------------------------------------- brute-force <D>


def _kepler(M, e):
    E = M.copy()
    for _ in range(40):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E


def brute_average(a, e, a_planet, k, k_planet, sigma, n_per_turn=2000):
    """<D>(sigma), D = 1/|r - r_P| - r.r_P / r_P^3, for a planar body (varpi = 0) and a circular planet; the angle is
    sigma = k M + k_P lambda_P (phase 0), so along one sigma M = (sigma - k_P lambda_P) / k and |k| turns of lambda_P
    close the line."""
    lam = np.linspace(0, TWO_PI * abs(k), n_per_turn * abs(k), endpoint=False)
    rp = np.stack([a_planet * np.cos(lam), a_planet * np.sin(lam)])
    out = []
    for s in np.atleast_1d(sigma):
        E = _kepler(np.mod((s - k_planet * lam) / k, TWO_PI), e)
        r = np.stack([a * (np.cos(E) - e), a * np.sqrt(1 - e * e) * np.sin(E)])
        out.append(np.mean(1 / np.hypot(*(r - rp)) - (r * rp).sum(0) / a_planet**3))
    return np.array(out)


def _axis(n_rad_yr):
    return (TWO_PI**2 / n_rad_yr**2) ** (1 / 3)


def _resonant_axes(spec, planet):
    n_planet = math.radians(const.PLANET_N_DEG_PER_GAUSS_YR[planet])
    k, k_planet = spec.body_coefficient, spec.planet_coefficients[0]
    n_star = -k_planet * n_planet / k  # sigma_dot = k n + k_P n_P = 0 (varpi constant)
    return _axis(n_star), _axis(n_planet), n_star


class Pendulum:
    """sigma in (3 k^2 / a^2) G m_P <D>(sigma - shift) of a circular coplanar planet (shift moves the potential)."""

    def __init__(self, key='2N-3', planet='Neptune', e=0.15, shift=0.0, n_sigma=360):
        self.spec = ResonanceSpec.from_key(key)
        self.e = e
        k, k_planet = self.spec.body_coefficient, self.spec.planet_coefficients[0]
        self.a_star, a_planet, self.n = _resonant_axes(self.spec, planet)
        self.c = -1.5 * k * self.n / self.a_star
        grid = np.linspace(0, TWO_PI, n_sigma, endpoint=False)
        values = brute_average(self.a_star, e, a_planet, k, k_planet, grid - shift)
        self.grid, self.values = grid, values
        spectrum = np.fft.rfft(values) / values.size
        self.m = np.arange(spectrum.size)[1:40]
        self.F = spectrum[1:40]
        self.coef = 3 * k * k / self.a_star**2 * TWO_PI**2 * const.PLANETS_MASS[planet]
        i0 = int(np.argmin(values))
        h = grid[1] - grid[0]
        curvature = (values[(i0 + 1) % n_sigma] - 2 * values[i0] + values[i0 - 1]) / h**2
        self.w0 = math.sqrt(self.coef * curvature)
        self.T0 = TWO_PI / self.w0
        self.v_sep = math.sqrt(2 * self.coef * (values.max() - values.min()))  # sigma_dot at the bottom on the separatrix
        self.sigma_min = grid[i0]

    def _rhs(self, t, y):
        slope = np.real(np.sum(2j * self.m * self.F * np.exp(1j * self.m * y[0])))
        return [y[1], -self.coef * slope]

    def _run(self, t, y0):
        return solve_ivp(self._rhs, (t[0], t[-1]), y0, t_eval=t, rtol=1e-10, atol=1e-12, method='DOP853').y

    def record(self, segments, step=20.0, sigma0=None):
        """segments [(t_end, sigma_dot)]: each later segment starts with sigma_dot at the first passage of the
        potential minimum after the previous t_end (an instantaneous kick of a there). Returns (series, kick times)."""
        sigma0 = self.sigma_min if sigma0 is None else sigma0
        times, angles, rates, kicks = [], [], [], []
        t0, s0 = 0.0, sigma0
        for j, (t_end, rate) in enumerate(segments):
            if j == len(segments) - 1:
                t = np.arange(t0, t_end + step / 2, step)
                y = self._run(t, [s0, rate])
                times.append(t)
                angles.append(y[0])
                rates.append(y[1])
                break
            t = np.arange(t0, t_end + 3 * self.T0, step)
            y = self._run(t, [s0, rate])
            d = np.mod(y[0] - self.sigma_min + np.pi, TWO_PI) - np.pi
            cross = np.where((t[:-1] >= t_end) & (np.sign(d[:-1]) != np.sign(d[1:])) & (np.abs(d[:-1]) < 1))[0][0] + 1
            times.append(t[:cross])
            angles.append(y[0][:cross])
            rates.append(y[1][:cross])
            t0, s0 = t[cross], y[0][cross]
            kicks.append(t0)
        t, sigma, rate = np.concatenate(times), np.concatenate(angles), np.concatenate(rates)
        zero = np.zeros_like(t)
        series = MMRSeries(
            t_yr=t,
            a=self.a_star + rate / self.c,
            e=zero + self.e,
            inc=zero + 1e-3,
            Omega=zero,
            omega=zero,
            varpi=zero,
            angle_wrapped=np.mod(sigma, TWO_PI),
            resonance=self.spec,
            regime=Regime(step_yr=step, span_yr=float(t[-1] - t[0]), n_rows=t.size, integrator='synthetic'),
            name='synthetic',
        )
        return series, kicks


@pytest.fixture(scope='module')
def plutino():
    return Pendulum('2N-3', 'Neptune', e=0.15)


# ------------------------------------------------------------------------------------- strength: the well rule


def _quadrature(settings):
    return strength_module.quadrature(settings)


def test_plutino_well_and_top_match_brute_force(settings):
    """2N-3 (sigma = 2 lambda_N - 3 lambda + varpi) at e = 0.15 (q = 33.5 au, no crossing): Plutinos librate about
    180 deg, the conjunction at aphelion, and the top of <D> is the conjunction at perihelion, 0 deg."""
    spec = ResonanceSpec.from_key('2N-3')
    a, a_planet, _ = _resonant_axes(spec, 'Neptune')
    k, k_planet = spec.body_coefficient, spec.planet_coefficients[0]
    body = ap.Ellipse(a, Orbit(0.15, 0.0, 0.0, 0.0))
    planet = ap.Ellipse(a_planet, Orbit(0.0, 0.0, 0.0, 0.0))
    h = 1e-3
    brute = brute_average(a, 0.15, a_planet, k, k_planet, [np.pi - h, np.pi, np.pi + h])
    curvature = (brute[0] - 2 * brute[1] + brute[2]) / h**2
    for observed in (np.pi, 2.4, 4.0):
        result = ap.centre_curvature(body, planet, k, k_planet, 0.0, observed, _quadrature(settings))
        assert result.measured
        assert ap.circular_distance(result.centre_rad, np.pi) < math.radians(0.5)
        assert ap.circular_distance(result.global_max_rad, 0.0) < math.radians(5.5)  # the coarse grid: 72 per turn
        assert result.curvature == pytest.approx(curvature, rel=0.01)
    centre = np.pi
    top = result.global_max_rad
    assert ap.swing_in_well(centre, top, np.pi - 1.0, np.pi + 1.2)
    assert not ap.swing_in_well(centre, top, np.pi - 1.0, TWO_PI + 0.3)  # passes the top at 2 pi
    assert not ap.swing_in_well(centre, top, -0.5, 0.7)  # a libration about the top, not about this centre


def test_exterior_one_to_two_horseshoe_counts_and_swing_over_the_top_does_not(settings):
    """1N-2 (sigma = lambda_N - 2 lambda + varpi) at e = 0.3: two asymmetric wells, an inner maximum at 180 deg and
    the global one at 0 deg (brute force here). A horseshoe 20..340 deg around both wells crosses only the inner
    maximum (Gallardo 2006, sect. 3.1) and counts; a swing through 0 deg does not."""
    spec = ResonanceSpec.from_key('1N-2')
    a, a_planet, _ = _resonant_axes(spec, 'Neptune')
    k, k_planet = spec.body_coefficient, spec.planet_coefficients[0]
    grid = np.radians(np.arange(0.0, 360.0, 0.5))
    values = brute_average(a, 0.3, a_planet, k, k_planet, grid, n_per_turn=1000)
    local_min = [i for i in range(grid.size) if values[i] < values[i - 1] and values[i] < values[(i + 1) % grid.size]]
    local_max = [i for i in range(grid.size) if values[i] > values[i - 1] and values[i] > values[(i + 1) % grid.size]]
    assert len(local_min) == 2 and len(local_max) == 2
    well = grid[local_min[0]]
    top = grid[max(local_max, key=lambda i: values[i])]
    inner = grid[min(local_max, key=lambda i: values[i])]
    assert 55 < math.degrees(well) < 90 and abs(math.degrees(well) + math.degrees(grid[local_min[1]]) - 360) < 1.0
    assert math.degrees(ap.circular_distance(top, 0.0)) < 1.0 and abs(math.degrees(inner) - 180.0) < 1.0

    body = ap.Ellipse(a, Orbit(0.3, 0.0, 0.0, 0.0))
    planet = ap.Ellipse(a_planet, Orbit(0.0, 0.0, 0.0, 0.0))
    result = ap.centre_curvature(body, planet, k, k_planet, 0.0, well + 0.2, _quadrature(settings))
    assert result.measured
    assert ap.circular_distance(result.centre_rad, well) < math.radians(1.0)
    assert ap.circular_distance(result.global_max_rad, top) < math.radians(5.5)

    deg = math.radians
    assert ap.swing_in_well(result.centre_rad, result.global_max_rad, deg(20), deg(340))  # horseshoe
    assert ap.swing_in_well(result.centre_rad, result.global_max_rad, deg(50), deg(100))  # tadpole in its well
    assert not ap.swing_in_well(result.centre_rad, result.global_max_rad, deg(-40), deg(100))  # over the top
    assert not ap.swing_in_well(result.centre_rad, result.global_max_rad, deg(100), deg(200))  # misses the centre
    # the same swings one turn up the unwrapped axis
    assert ap.swing_in_well(result.centre_rad, result.global_max_rad, deg(380), deg(700))
    assert not ap.swing_in_well(result.centre_rad, result.global_max_rad, deg(320), deg(460))


def test_swing_in_well_hand_cases():
    """<D> = -cos sigma: centre 0, top pi. Angles are compared on the unwrapped axis nearest the swing's middle."""
    assert ap.swing_in_well(0.0, np.pi, -1.0, 1.0)
    assert ap.swing_in_well(TWO_PI, np.pi, -1.0, 1.0)  # the same centre given one turn up
    assert ap.swing_in_well(0.0, np.pi, TWO_PI - 1.0, TWO_PI + 1.0)
    assert ap.swing_in_well(0.0, -np.pi, -3.0, 3.0)  # the top is on the edge, not strictly inside
    assert not ap.swing_in_well(0.0, np.pi, -1.0, 3.5)  # passes pi
    assert not ap.swing_in_well(0.0, np.pi, 1.0, 2.0)  # does not hold the centre
    assert not ap.swing_in_well(float('nan'), np.pi, -1.0, 1.0)
    assert not ap.swing_in_well(0.0, float('nan'), -1.0, 1.0)


def test_observed_swing_takes_the_half_cycle_of_observed_centre(plutino, settings):
    series, _ = plutino.record([(100000, 0.6 * plutino.w0)])
    episode = next(run for run in analyse_mmr(series, settings).geometry.runs if run.accepted)
    episode = dataclasses.replace(episode, tp_times_yr=np.array([0.0, 100.0, 200.0, 300.0]), tp_values_rad=np.array([1.0, 3.0, 1.2, 2.8]))
    assert strength_module.observed_swing(episode, 150.0) == pytest.approx((1.2, 3.0))
    assert strength_module.observed_swing(episode, 50.0) == pytest.approx((1.0, 3.0))
    assert strength_module.observed_centre(episode, 150.0) == pytest.approx(2.1)
    single = dataclasses.replace(episode, tp_times_yr=np.array([0.0]), tp_values_rad=np.array([1.0]))
    assert all(math.isnan(x) for x in strength_module.observed_swing(single, 0.0))


def test_strength_of_a_true_pendulum_is_consistent(plutino, settings):
    """A pendulum in the true <R>: every evaluation is in the well, required/theory ~ 1 (the period here comes from the
    brute-force curvature: T0 = 2 pi / (|k| n sqrt(3 mu_P a <D>'')))."""
    k = plutino.spec.body_coefficient
    mu_eff = plutino.w0**2 / (3 * k * k * plutino.n**2)
    assert TWO_PI / (abs(k) * plutino.n * math.sqrt(3 * mu_eff)) == pytest.approx(plutino.T0)
    series, _ = plutino.record([(200000, 0.6 * plutino.w0)])
    analysis = analyse_mmr(series, settings)
    s = analysis.strength
    assert analysis.verdict.angle_status == AngleStatus.LIBRATION
    assert s.method == strength_module.METHOD_DIRECT
    assert s.check == CheckValue.PASS
    assert s.n_evaluations > 0 and s.n_consistent == s.n_evaluations
    assert 0.8 < s.ratio < 1.25
    assert s.not_measured_reason == ''
    columns = list(strength_module.strength_columns(s))
    assert columns.index('str_n_consistent') == columns.index('str_n_evaluations') + 1


def test_libration_about_the_top_of_the_true_potential_fails_strength(plutino, settings):
    """The 482824 situation built on purpose: the body librates about 0 deg (where the true <D> of 2N-3 has its top)
    in a well of the same curvature. The ratio agrees (~1) by accident; no half-cycle holds the minimum at 180 deg,
    so no evaluation is consistent and F fails with centre_outside_well, the ratio staying a column."""
    moved = Pendulum('2N-3', 'Neptune', e=0.15, shift=np.pi)
    assert ap.circular_distance(moved.sigma_min, 0.0) < 1e-9
    series, _ = moved.record([(200000, 0.6 * moved.w0)])
    analysis = analyse_mmr(series, settings)
    s = analysis.strength
    assert analysis.verdict.angle_status == AngleStatus.LIBRATION
    assert s.check == CheckValue.FAIL
    assert s.not_measured_reason == strength_module.REASON_CENTRE_OUTSIDE_WELL
    assert s.n_consistent == 0 and s.n_evaluations > 0
    assert 0.8 < s.ratio < 1.25  # the accidental agreement, kept as a column
    assert not s.implausible
    assert analysis.verdict.checks['check_mmr_strength'] == CheckValue.FAIL
    assert abs(s.centre_offset_max_deg - 180.0) < 5.0


def test_horseshoe_of_an_exterior_one_to_two_counts(settings):
    """A pendulum in the true 1N-2 potential (e = 0.3) energetic enough to go round both asymmetric wells over the
    inner maximum (a horseshoe, swing ~20..340 deg) and a small libration in one well: both are consistent."""
    p = Pendulum('1N-2', 'Neptune', e=0.3)
    inner = p.values[p.grid.size // 2]
    v_inner = math.sqrt(2 * p.coef * (inner - p.values.min()))
    for factor, horseshoe in ((0.5, False), (1.3, True)):
        series, _ = p.record([(300000, factor * v_inner)])
        unwrapped = np.unwrap(series.angle_wrapped)
        assert (np.ptp(unwrapped) > np.pi) == horseshoe
        assert unwrapped.min() > 0 and unwrapped.max() < TWO_PI  # never over the global top at 0
        s = analyse_mmr(series, settings).strength
        assert s.n_consistent > 0, factor
        assert s.check == CheckValue.PASS, factor


# ------------------------------------------------------------------------------------- decide: G1l, rule C, L4


def _confirmed_jumps(geometry):
    return [jump for jump in geometry.jumps if not jump.kick]


def test_capture_after_a_jump_is_transient_ending_in_libration(plutino, settings):
    """Circulation (sigma_dot = 1.5 x the separatrix value) to a kick of a at ~60 kyr, then 140 kyr of libration
    (sigma_dot 0.8 omega0 at the minimum): transient, the angle episodic over the record, final_regime libration
    from the last episode."""
    series, kicks = plutino.record([(60000, 1.5 * plutino.v_sep), (200000, 0.8 * plutino.w0)])
    analysis = analyse_mmr(series, settings)
    v = analysis.verdict
    jumps = _confirmed_jumps(analysis.geometry)
    assert jumps and abs(jumps[-1].time_yr - kicks[0]) < 2000.0
    assert v.angle_status == AngleStatus.EPISODIC
    assert v.status == Status.TRANSIENT
    assert v.final_regime == AngleStatus.LIBRATION
    assert kicks[0] <= v.final_regime_start_yr <= kicks[0] + plutino.T0


def test_capture_after_a_fast_circulation_is_transient_ending_in_libration(plutino, settings):
    """As above with a faster circulation before the jump (3 x the separatrix rate) and 15 libration cycles after
    it: the spec (G1l) gives transient with final_regime libration."""
    series, kicks = plutino.record([(60000, 3.0 * plutino.v_sep), (400000, 0.8 * plutino.w0)])
    analysis = analyse_mmr(series, settings)
    v = analysis.verdict
    window = analysis.geometry.window
    after = np.unwrap(series.angle_wrapped)[series.t_yr > window[0]]
    assert np.ptp(after) < np.pi  # the record itself librates after the jump
    assert v.angle_status == AngleStatus.EPISODIC and v.status == Status.TRANSIENT
    assert v.final_regime == AngleStatus.LIBRATION


def test_libration_on_both_sides_of_a_jump_stays_libration(plutino, settings):
    """A large libration (2 omega0 at the minimum) kicked to a small one at ~100 kyr: the earliest librating episode
    covers the start and reaches the jump, so G1l does not fire."""
    series, kicks = plutino.record([(100000, 2.0 * plutino.w0), (300000, 0.3 * plutino.w0)])
    analysis = analyse_mmr(series, settings)
    v = analysis.verdict
    jumps = _confirmed_jumps(analysis.geometry)
    assert jumps and abs(jumps[-1].time_yr - kicks[0]) < 2000.0
    assert v.angle_status == AngleStatus.LIBRATION
    assert v.status == Status.RESONANT
    assert any(entry.name.startswith('G1l') for entry in v.decision_path)


def _member(key):
    spec = ResonanceSpec.from_key(key)
    nan = float('nan')
    return SubresonanceCandidate(
        spec=spec, j=2, nodal=False, centre_range_rad=0.1, centre_rate_rad_yr=0.0, centre_max_pass_rad=0.1,
        centre_mean_rad=np.pi, phi0_coherence=0.98, coefficient_rel=1.0, composite_rms_rad=nan, period_corr=nan,
        period_ratio=1.0, rank=0, accepted=True, reason='',
    )  # fmt: skip


def _subresonances(member):
    nan = float('nan')
    return SubresonanceResult(
        triggers=(), support=(), candidates=[member] if member else [], accepted=member,
        check=CheckValue.FAIL if member else CheckValue.PASS, composite=False, undetermined=False, tie_by_order=False,
        theory_measured=False, sigma0_phi0_coherence=nan, composite_rms_rad=nan, single_rms_rad=nan, binomial_mode=0,
        drift_pred_rad_yr=nan, drift_meas_rad_yr=nan, plane_offset_rad=nan,
    )  # fmt: skip


def _verdict(status, angle_status):
    return Verdict(resonance_type=ResonanceType.MMR, status=status, angle_status=angle_status)


@pytest.mark.parametrize(
    'angle_status, member_status_, member_angle, expected',
    [
        (AngleStatus.CIRCULATION, Status.TRANSIENT, AngleStatus.EPISODIC, Status.TRANSIENT),  # 82831 13M-24
        (AngleStatus.CIRCULATION, Status.RESONANT, AngleStatus.LIBRATION, Status.RESONANT),
        (AngleStatus.EPISODIC, Status.RESONANT, AngleStatus.LIBRATION, Status.RESONANT),  # not over the record
        (AngleStatus.CIRCULATION, Status.INVALID, AngleStatus.UNDEFINED, Status.UNCERTAIN),
        (AngleStatus.DRIFT, Status.NON_RESONANT, AngleStatus.CIRCULATION, Status.NON_RESONANT),
    ],
)
def test_catalogue_row_takes_the_member_status(settings, angle_status, member_status_, member_angle, expected):
    """Rule C: the members of one commensurability are one resonance; the catalogue row takes the accepted member's
    status (invalid -> uncertain), reason subresonance_librates, origin subresonance, owner = the member, rule 1."""
    member = _member('13M-24+2+9')
    angle = AngleDecision(angle_status=angle_status, subtype=AngleSubtype.NONE, rule='G4')
    decision = member_status(angle, _subresonances(member), _verdict(member_status_, member_angle), Ledger.from_settings(settings))
    assert decision is not None
    assert decision.status == expected
    assert decision.reason == Reason.SUBRESONANCE_LIBRATES
    assert decision.origin == LibrationOrigin.SUBRESONANCE
    assert decision.owner == '13M-24+2+9'
    assert decision.rule == '1'


def test_both_angles_librating_keep_their_own_verdicts(settings):
    """Apsidal corotation (Beauge et al. 2003): the catalogue angle and the member both librate over the record, so the
    catalogue row keeps its own verdict (None) and members_locked = 1; without a member row nothing is taken."""
    member = _member('13M-24+2+9')
    angle = AngleDecision(angle_status=AngleStatus.LIBRATION, subtype=AngleSubtype.STATIONARY, rule='G2', episode_id=0)
    locked = _verdict(Status.RESONANT, AngleStatus.LIBRATION)
    ledger = Ledger.from_settings(settings)
    assert member_status(angle, _subresonances(member), locked, ledger) is None
    assert members_locked(angle, locked) == 1
    circulating = AngleDecision(angle_status=AngleStatus.CIRCULATION, subtype=AngleSubtype.NONE, rule='G4')
    assert math.isnan(members_locked(circulating, locked))
    assert member_status(circulating, _subresonances(member), None, ledger) is None
    assert member_status(circulating, _subresonances(None), locked, ledger) is None


def _force_record(phase_of_t, noise=0.0, seed=1):
    period, amplitude = 1000.0, 1.0
    t = np.arange(0.0, 40 * period, 5.0)
    sigma = np.pi + amplitude * np.sin(TWO_PI * t / period)
    adot = 2e-6 * np.sin(sigma - phase_of_t(t)) + 3e-8
    if noise:
        adot = np.random.default_rng(seed).normal(0.0, noise, t.size)
    return t, sigma, adot, period, amplitude


def test_force_law_with_a_moving_phase_passes_l4(settings):
    """A clean law a_dot = R sin(sigma - phi0) whose phase turns by 120 deg over the record (each window R^2 ~ 1): the
    phi0 spread no longer decides (W7 'phi0' variant D), so L4 passes; the windows' phases follow phi0(t)."""
    phase = lambda t: math.radians(120.0) * t / t[-1] + 0.4  # noqa: E731
    t, sigma, adot, period, amplitude = _force_record(phase)
    windows = force_windows(t, sigma, adot, settings.mmr_force_window_cycles * period, settings.mmr_force_window_step)
    assert len(windows) >= settings.mmr_force_min_windows
    phases = np.array([w.phi0_rad for w in windows])
    mids = np.array([(w.t_start_yr + w.t_end_yr) / 2 for w in windows])
    expected = math.radians(120.0) * mids / t[-1] + 0.4
    assert np.max(np.abs(np.angle(np.exp(1j * (phases - expected))))) < math.radians(3.0)
    spread = np.max(np.abs(np.angle(np.exp(1j * (phases - np.angle(np.mean(np.exp(1j * phases))))))))
    assert spread > math.radians(settings.mmr_force_phase_tol_deg)  # the old L4 entry would have refused it
    assert min(w.r2 for w in windows) > 0.95
    check, thin, _ = a_response_check(windows, None, amplitude, settings)
    assert check == CheckValue.PASS and not thin


def test_force_law_of_noise_fails_l4(settings):
    t, sigma, adot, period, amplitude = _force_record(lambda t: 0.0 * t, noise=1e-6)
    windows = force_windows(t, sigma, adot, settings.mmr_force_window_cycles * period, settings.mmr_force_window_step)
    check, _, _ = a_response_check(windows, None, amplitude, settings)
    assert check == CheckValue.FAIL


# ------------------------------------------------------------------------------------- geometry: slow edges

HALF = 643.0  # median half-period [yr] (2005 GQ107 40M-57)
AMP = 0.6  # half-amplitude [rad]


def _turning_points(durations, drift, last_extra=0.0):
    """TPs alternating max/min with half-amplitude AMP whose midpoints drift by `drift` per half-cycle; `durations` are
    the half-cycle lengths [yr]. The bands of a run of N TPs then share a point while 2 A - (N - 3) |b| > 0."""
    times = 1000.0 + np.concatenate([[0.0], np.cumsum(durations)])
    n = times.size
    values = np.array([drift * i + AMP * (-1) ** i for i in range(n)])
    values[-1] += last_extra * (-1) ** (n - 1)
    kinds = np.array([1 if i % 2 == 0 else -1 for i in range(n)])
    tp = TurningPoints(indices=times.astype(int), times_yr=times, values_rad=values, kinds=kinds, prominence_rad=0.02)
    t = np.arange(0.0, times[-1] + 3000.0, 1.0)
    return tp, t, (0.0, float(t[-1]))


def _summary(episodes):
    return [(e.episode_id, e.accepted, str(e.reject_reason), len(e.tp_times_yr)) for e in episodes]


def test_slow_last_half_cycle_is_cut_off_the_episode(settings):
    """GQ107: 12 regular TPs, a 13th after a half-cycle 2.07 x the median (its full period only (1 + 2.07)/2 = 1.54 x).
    With b = 0.127 rad per half-cycle: 13 TPs give 1.2 - 10 b < 0 (centre moves), 12 give 1.2 - 9 b > 0 (stationary)."""
    tp, t, window = _turning_points([HALF] * 11 + [2.07 * HALF], drift=0.127, last_extra=1.0)
    whole = classify_run(0, tp, np.arange(13), t, window, settings)
    assert not whole.accepted and str(whole.reject_reason) == 'centre_moves'
    episodes = find_episodes(t, tp, window, settings)
    assert _summary(episodes) == [(0, True, '', 12), (1, False, 'edge', 1)]
    assert episodes[0].tp_times_yr[-1] == pytest.approx(tp.times_yr[11])


def test_edge_half_cycle_not_slow_enough_is_kept(settings):
    """1.9 x the median half-period is not slow (> 2 x needed): nothing is cut, the run stays rejected."""
    tp, t, window = _turning_points([HALF] * 11 + [1.9 * HALF], drift=0.127)
    assert _summary(find_episodes(t, tp, window, settings)) == [(0, False, 'centre_moves', 13)]


def test_uniform_drift_is_not_accepted_on_a_shorter_piece(settings):
    """A uniform drift b = 0.2 with slow half-cycles at both ends: one cut per side leaves 11 TPs, 1.2 - 8 b < 0; a
    5-TP piece of the same drift would pass (1.2 - 2 b > 0), so only a single cut per side keeps the drift out."""
    tp, t, window = _turning_points([2.5 * HALF] + [HALF] * 10 + [2.5 * HALF], drift=0.2)
    assert classify_run(0, tp, np.arange(4, 9), t, window, settings).accepted  # what an iterated trim would reach
    episodes = find_episodes(t, tp, window, settings)
    assert not any(e.accepted for e in episodes)


def test_slow_first_half_cycle_is_cut_off_the_episode(settings):
    """The mirror image: the first half-cycle 2.07 x slower; the rejected `edge` piece keeps the run's id, the core
    the next one (consecutive ids)."""
    tp, t, window = _turning_points([2.07 * HALF] + [HALF] * 11, drift=0.127)
    episodes = find_episodes(t, tp, window, settings)
    assert _summary(episodes) == [(0, False, 'edge', 1), (1, True, '', 12)]
    assert episodes[1].tp_times_yr[0] == pytest.approx(tp.times_yr[1])


def test_slow_half_cycles_at_both_ends_are_cut(settings):
    """b = 0.142: 13 TPs (1.2 - 10 b) and either 12-TP cut (1.2 - 9 b) still drift, the 11-TP core (1.2 - 8 b = 0.064)
    is stationary: both edges go."""
    tp, t, window = _turning_points([2.5 * HALF] + [HALF] * 10 + [2.5 * HALF], drift=0.142)
    assert _summary(find_episodes(t, tp, window, settings)) == [(0, False, 'edge', 1), (1, True, '', 11), (2, False, 'edge', 1)]


def test_accepted_run_is_never_trimmed(settings):
    tp, t, window = _turning_points([HALF] * 11 + [2.07 * HALF], drift=0.0, last_extra=1.0)
    assert _summary(find_episodes(t, tp, window, settings)) == [(0, True, '', 13)]


# ------------------------------------------------------------------------------------- planets


@pytest.mark.parametrize(
    'key, expected_deg',
    [
        ('5J+6S-1', abs(5 * 0.34 - 6 * 0.83)),  # same sign: the antiphase waves partly cancel, 3.28 deg
        ('5J-4S-1', 5 * 0.34 + 4 * 0.83),  # opposite sign: they add, 5.02 deg
        ('2J-1', 2 * 0.34),  # Jupiter alone
        ('3S-1', 3 * 0.83),  # Saturn alone
    ],
)
def test_great_inequality_ripple_by_hand(key, expected_deg):
    """r_GI = |k_J dlambda_J - k_S dlambda_S| G(P_GI): dlambda_J = +A_J sin theta, dlambda_S = -A_S sin theta. Analog
    Butterworth G = 1 / (1 + (P_c / P)^(2N)): P_GI = 900 yr, P_c = 300 yr, N = 2 -> 1 / (1 + 1/81) = 81/82."""
    gi = GreatInequality(period_yr=900.0, dlambda_jupiter_rad=math.radians(0.34), dlambda_saturn_rad=math.radians(0.83), measured=False)
    ripple = predicted_ripple(ResonanceSpec.from_key(key), gi, 300.0, 2, None)
    assert math.degrees(ripple) == pytest.approx(expected_deg * 81 / 82, rel=1e-9)


def test_catalogue_phase_error_bound_by_hand():
    """D = sum |k_j| min(SPREAD_j T, 2 SLOW_MAX_j + OFFSET_j T) with the W7 tables (deg): Jupiter 1.5e-4, 4.1e-6, 1.91;
    Saturn 3.7e-4, 8.5e-6, 4.69. T = 10 kyr: J min(1.5, 3.861) = 1.5, S min(3.7, 9.465) = 3.7 -> 5 x 1.5 + 6 x 3.7
    = 29.7 deg. T = 100 kyr: J min(15, 4.23) = 4.23, S min(37, 10.23) = 10.23 -> 82.53 deg."""
    spec = ResonanceSpec.from_key('5J+6S-1')
    assert math.degrees(catalogue_phase_error(spec, ('Jupiter', 'Saturn'), 1e4)) == pytest.approx(29.7, abs=1e-6)
    assert math.degrees(catalogue_phase_error(spec, ('Jupiter', 'Saturn'), 1e5)) == pytest.approx(82.53, abs=1e-6)
    assert math.degrees(catalogue_phase_error(spec, ('Jupiter',), 1e5)) == pytest.approx(21.15, abs=1e-6)
    assert catalogue_phase_error(spec, (), 1e5) == 0.0


def test_coherence_tolerance_constant_bounds_the_sinc_loss():
    """Window phases spread uniformly over D have coherence |sinc(D/2)|; 0.2181 is the largest slope of sinc(x/2), so
    the loss 1 - sinc(D/2) never exceeds 0.2181 D."""
    x = np.linspace(1e-6, 40.0, 400001)
    slope = np.max(np.abs(np.gradient(np.sin(x / 2) / (x / 2), x)))
    assert SINC_HALF_SLOPE_MAX == pytest.approx(slope, abs=2e-4)
    for spread in (0.05, 0.5, 1.1, 2.0, 4.0):
        phases = np.linspace(-spread / 2, spread / 2, 2001)
        coherence = abs(np.mean(np.exp(1j * phases)))
        assert 1 - coherence <= SINC_HALF_SLOPE_MAX * spread + 1e-9


def test_catalogue_longitudes_carry_the_table_rate():
    """lambda_j(t_start) = lambda_j(t_ref) + n_j (t_start - t_ref) mod 2 pi; Mercury is never catalogued. By hand for
    1000 yr: Jupiter 30349.646 deg = 84 turns + 109.646 deg; Earth 360000.505 deg = 1000 turns + 0.505 deg."""
    assert 'Mercury' not in CATALOGUE_PLANETS
    out = catalogue_longitudes({'Jupiter': 1.0, 'Earth': 6.0, 'Mercury': 0.5}, 0.0, 1000.0)
    assert set(out) == {'Jupiter', 'Earth'}
    assert out['Jupiter'] == pytest.approx(1.0 + math.radians(109.646), abs=1e-9)
    assert out['Earth'] == pytest.approx(6.0 + math.radians(0.505), abs=1e-9)
