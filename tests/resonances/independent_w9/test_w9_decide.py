"""Independent functional tests of the W9 changes (decide G1l, rule C, L4 variant D, the kinematic episode, B5 by
geometry, the well of the two-body strength, the slow edge half-cycle, the C11 ripple, catalogue longitudes).

Every expected value is derived here from the physics or by hand, not read from the implementation:

- strength: <D>(sigma) = -cos(sigma - c) has its minimum at c and its maximum at c + pi; a swing inside the well
  contains c and stays below the top, a swing over the top contains c + pi;
- decide: synthetic pendulum records sigma_ddot = -w0^2 sin(sigma - pi) with the body's lambda, n and a built from
  sigma (n = 2 n_J - sigma_dot for 2J-1), so the identity and the force law hold by construction;
- geometry: turning points of a libration of half-amplitude A whose centre drifts by b per half-cycle; the band test
  of a uniform drift over N turning points has the margin 2A - (N - 3) b (docstring of `_trim_slow_edges`);
- planets: the great-inequality waves of Jupiter and Saturn are in antiphase, so the wave of k_J lambda_J + k_S
  lambda_S has the half-amplitude |k_J A_J - k_S A_S|;
- the catalogue error bound D = sum_j |k_j| min(SPREAD_j T, 2 SLOW_MAX_j + OFFSET_j T) with the tables of const.
"""

import dataclasses
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from resonances.data import const
from resonances.resonance.analysis import geometry as geo
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import decide, owner, response
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.averaged_potential import basin_minimum, swing_in_well
from resonances.resonance.analysis.mmr.context import catalogue_longitudes
from resonances.resonance.analysis.mmr.models import (
    GreatInequality,
    Kinematics,
    MMRSeries,
    MultipletFit,
    PlanetSeries,
    Regime,
    ResonanceSpec,
    Response,
    SubresonanceCandidate,
    SubresonanceResult,
)
from resonances.resonance.analysis.mmr.preprocess import predicted_ripple
from resonances.resonance.analysis.mmr.strength import observed_swing
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import (
    AngleStatus,
    AngleSubtype,
    CheckValue,
    Reason,
    ResonanceType,
    Status,
    Verdict,
)

SETTINGS = AnalysisSettings()
N_J = math.radians(const.PLANET_N_DEG_PER_GAUSS_YR["Jupiter"])
P0_YR = 600.0
W0 = 2 * np.pi / P0_YR


_EMPTY = {
    'float': lambda: float('nan'),
    'int': lambda: 0,
    'bool': lambda: False,
    'str': lambda: '',
    'CheckValue': lambda: CheckValue.NOT_MEASURED,
    'np.ndarray': lambda: np.array([]),
}


def _empty(kind: str):
    if kind.startswith('List'):
        return []
    if kind.startswith('Tuple'):
        return ()
    return _EMPTY.get(kind, lambda: None)()


def blank(cls, **values):
    """An instance of a dataclass with every field not in `values` set to 'not measured' (nan, None, empty)."""
    kwargs = dict(values)
    for f in dataclasses.fields(cls):
        has_default = f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
        if f.name not in values and not has_default:
            kwargs[f.name] = _empty(str(f.type))
    return cls(**kwargs)


# --------------------------------------------------------------------------------------------- strength: the well


@pytest.mark.parametrize('centre_deg', [40.0, 180.0, 350.0])
def test_swing_in_well_of_a_cosine_potential(centre_deg):
    """<D> = -cos(sigma - c): the minimum (stable centre) at c, the global maximum at c + pi."""
    c = math.radians(centre_deg)
    grid = np.linspace(0.0, 2 * np.pi, 360, endpoint=False)
    values = -np.cos(grid - c)
    top = (c + np.pi) % (2 * np.pi)
    # the steepest descent from any point not on the top ends in the one minimum at c
    for start in (0, 90, 200, 300):
        if abs(math.remainder(grid[start] - top, 2 * np.pi)) > 0.1:
            assert basin_minimum(values, start) == int(round(centre_deg)) % 360
    # a swing of +-60 deg about the centre, on an unwrapped branch two turns up: inside the well
    branch = 4 * np.pi
    assert swing_in_well(c, top, c + branch - math.radians(60), c + branch + math.radians(60))
    # a swing of +-170 deg still stays below the top
    assert swing_in_well(c, top, c - math.radians(170), c + math.radians(170))
    # a swing that holds the centre and runs over the top (c - 30 deg .. c + 200 deg)
    assert not swing_in_well(c, top, c - math.radians(30), c + math.radians(200))
    # a swing that does not hold the centre (c + 20 .. c + 90 deg)
    assert not swing_in_well(c, top, c + math.radians(20), c + math.radians(90))
    # nothing finite to compare with
    assert not swing_in_well(float('nan'), top, c - 0.5, c + 0.5)
    assert not swing_in_well(c, float('nan'), c - 0.5, c + 0.5)


def test_observed_swing_is_the_sorted_half_cycle():
    tp_t = np.array([0.0, 100.0, 200.0, 300.0])
    tp_v = np.array([1.0, -1.0, 1.2, -0.8])
    episode = blank(geo.Episode, episode_id=0, tp_times_yr=tp_t, tp_values_rad=tp_v, accepted=True)
    # t = 250 yr lies inside the half-cycle 200..300 yr and at its middle
    lo, hi = observed_swing(episode, 250.0)
    assert (lo, hi) == pytest.approx((-0.8, 1.2))
    one = blank(geo.Episode, episode_id=0, tp_times_yr=tp_t[:1], tp_values_rad=tp_v[:1], accepted=True)
    assert all(math.isnan(x) for x in observed_swing(one, 0.0))


# ---------------------------------------------------------------------------------- synthetic pendulum records


def _pendulum(t, sigma0, rate0, centre=np.pi):
    sol = solve_ivp(
        lambda _, y: [y[1], -(W0**2) * np.sin(y[0] - centre)],
        (t[0], t[-1]),
        [sigma0, rate0],
        t_eval=t,
        rtol=1e-10,
        atol=1e-12,
        max_step=t[1] - t[0],
    )
    return sol.y[0], sol.y[1]


def capture_record(energy_before, t_jump_min, span=20000.0, step=1.0, energy_after=-0.5):
    """2J-1: a circulation of energy `energy_before` w0^2, then at the first passage of the centre (pi) after
    `t_jump_min` the rate drops to the libration of energy `energy_after` w0^2 (half-amplitude 60 deg): a jump of a."""
    t = np.arange(0.0, span + step / 2, step)
    sigma, rate = _pendulum(t, np.pi, W0 * math.sqrt(2 * (energy_before + 1)))
    near_centre = np.abs(np.angle(np.exp(1j * (sigma - np.pi)))) < 0.05
    i = int(np.argmax((t >= t_jump_min) & near_centre))
    rate_after = np.sign(rate[i]) * W0 * math.sqrt(2 * (energy_after + math.cos(sigma[i] - np.pi)))
    s2, r2 = _pendulum(t[i:], sigma[i], rate_after)
    sigma = np.concatenate([sigma[:i], s2])
    rate = np.concatenate([rate[:i], r2])
    varpi = 0.7
    lam_j = 0.3 + N_J * t
    lam = 2 * lam_j - varpi - sigma  # sigma = 2 lambda_J - lambda - varpi
    n = 2 * N_J - rate
    a = (4 * np.pi**2 / n**2) ** (1.0 / 3.0)
    one = np.ones_like(t)
    series = MMRSeries(
        t_yr=t,
        a=a,
        e=0.15 * one,
        inc=0.05 * one,
        Omega=1.1 * one,
        omega=(varpi - 1.1) * one,
        varpi=varpi * one,
        angle_wrapped=np.mod(sigma, 2 * np.pi),
        resonance=ResonanceSpec.from_key('2J-1'),
        regime=Regime(step_yr=step, span_yr=span, n_rows=len(t), integrator='ias15'),
        mean_longitude=np.mod(lam, 2 * np.pi),
        mean_anomaly=np.mod(lam - varpi, 2 * np.pi),
        name='capture',
    )
    return series, t[i], sigma


def _rules(verdict):
    return [entry.name for entry in verdict.decision_path]


@pytest.mark.xfail(
    strict=True,
    reason=(
        'W16: at tau = 2 P_c = 150 yr the robust spread of the a-increments holds the libration swing of a; '
        'the capture steps a by 0.083 au over tau against a threshold of 0.157 au and is not a jump candidate'
        ' (the verdict stays transient/libration, confidence low). TEST-DISPUTES W16'
    ),
)
def test_capture_after_a_jump_is_transient_with_final_regime_libration():
    """Circulation (sigma_dot ~ 4 w0) to ~6 kyr, then a libration of 60 deg to the end: G1l."""
    series, t_jump, _ = capture_record(8.0, 6000.0)
    analysis = analyse_mmr(series, SETTINGS)
    v = analysis.verdict
    assert len(analysis.geometry.jumps) >= 1
    assert abs(analysis.geometry.jumps[-1].time_yr - t_jump) < 2 * P0_YR
    assert any(name.startswith('G1l') for name in _rules(v))
    assert v.status == Status.TRANSIENT
    assert v.angle_status == AngleStatus.EPISODIC
    assert v.angle_subtype != AngleSubtype.RECURRENT  # one episode only, after the jump
    assert v.final_regime == AngleStatus.LIBRATION
    # the libration starts within a period of the jump
    assert t_jump <= v.final_regime_start_yr <= t_jump + 2 * P0_YR
    # no librating episode before the jump
    assert analysis.extra_columns['geo_n_episodes_before_jump'] == 0


@pytest.mark.slow
def test_capture_after_a_long_fast_circulation_is_transient():
    series, t_jump, _ = capture_record(20.0, 12000.0)
    v = analyse_mmr(series, SETTINGS).verdict
    assert v.status == Status.TRANSIENT
    assert v.angle_status == AngleStatus.EPISODIC
    assert v.final_regime == AngleStatus.LIBRATION


@pytest.mark.slow
def test_centre_of_a_capture_stays_inside_the_swing():
    """A low-pass of a libration bounded in [pi - 60 deg, pi + 60 deg] (one branch) stays near that band."""
    series, t_jump, sigma = capture_record(20.0, 12000.0)
    analysis = analyse_mmr(series, SETTINGS)
    centre = analysis.geometry.centre
    after = sigma[series.t_yr >= t_jump + 2 * P0_YR]
    inside = centre.t_yr >= t_jump + 2 * P0_YR
    lo, hi = after.min(), after.max()
    assert np.all(centre.centre_rad[inside] >= lo - 0.5)
    assert np.all(centre.centre_rad[inside] <= hi + 0.5)


# ------------------------------------------------------------------------------------------------- rule C


def member_record(span=20000.0, step=1.0, varpi_period=5000.0):
    """2J-1 = 2 lambda_J - lambda - varpi circulates while its member 2 lambda_J - lambda - varpi_J (2J-1-1+0 in the
    long notation: Jupiter's varpi) librates about pi with 60 deg: varpi turns once in `varpi_period`, varpi_J
    stands still. Both rows belong to one commensurability."""
    t = np.arange(0.0, span + step / 2, step)
    member, rate = _pendulum(t, np.pi, W0 * math.sqrt(2 * (-0.5 + 1)))
    varpi_j = 0.4
    lam_j = 0.3 + N_J * t
    varpi = 0.7 + 2 * np.pi / varpi_period * t
    lam = 2 * lam_j - varpi_j - member
    n = 2 * N_J - rate
    a = (4 * np.pi**2 / n**2) ** (1.0 / 3.0)
    one = np.ones_like(t)
    jupiter = PlanetSeries(
        name='Jupiter',
        t_yr=t,
        a=5.2026 * one,
        e=0.0484 * one,
        inc=0.0228 * one,
        Omega=1.75 * one,
        varpi=varpi_j * one,
        mean_longitude=np.mod(lam_j, 2 * np.pi),
    )
    return MMRSeries(
        t_yr=t,
        a=a,
        e=0.15 * one,
        inc=0.05 * one,
        Omega=1.1 * one,
        omega=np.mod(varpi - 1.1, 2 * np.pi),
        varpi=np.mod(varpi, 2 * np.pi),
        angle_wrapped=np.mod(2 * lam_j - lam - varpi, 2 * np.pi),
        resonance=ResonanceSpec.from_key('2J-1'),
        regime=Regime(step_yr=step, span_yr=span, n_rows=len(t), integrator='ias15', planets_saved=('Jupiter',)),
        planets={'Jupiter': jupiter},
        mean_longitude=np.mod(lam, 2 * np.pi),
        mean_anomaly=np.mod(lam - varpi, 2 * np.pi),
        name='member',
    )


@pytest.mark.slow
def test_catalogue_row_takes_the_status_of_its_librating_member():
    analysis = analyse_mmr(member_record(), SETTINGS)
    v = analysis.verdict
    assert analysis.subresonances is not None and analysis.subresonances.accepted is not None
    member_key = analysis.subresonances.accepted.spec.key
    assert ResonanceSpec.from_key(member_key).planet_varpi_coefficients == (-1,)
    assert ResonanceSpec.from_key(member_key).body_varpi_coefficient == 0
    derived = [d for d in analysis.derived if d.resonance.key == member_key]
    assert len(derived) == 1
    dv = derived[0].verdict
    assert dv.status == Status.RESONANT
    assert dv.angle_status == AngleStatus.LIBRATION
    # the catalogue row: the member's status and confidence, its own angle_status
    assert v.status == dv.status
    assert v.confidence == dv.confidence
    assert v.reason == Reason.SUBRESONANCE_LIBRATES
    assert v.owner == member_key
    assert v.angle_status == AngleStatus.CIRCULATION


def _sub_result(member_key):
    candidate = blank(SubresonanceCandidate, spec=ResonanceSpec.from_key(member_key), j=1, nodal=False, accepted=True)
    return blank(SubresonanceResult, candidates=[candidate], accepted=candidate, check=CheckValue.FAIL)


def _member_verdict(status, angle_status):
    return Verdict(resonance_type=ResonanceType.MMR, status=status, angle_status=angle_status)


def test_member_status_invalid_member_leaves_uncertain():
    angle = decide.AngleDecision(AngleStatus.CIRCULATION, AngleSubtype.NONE, 'G4')
    member = _member_verdict(Status.INVALID, AngleStatus.INVALID if hasattr(AngleStatus, 'INVALID') else '')
    out = decide.member_status(angle, _sub_result('2J-1-1+0'), member, Ledger.from_settings(SETTINGS))
    assert out is not None
    assert out.status == Status.UNCERTAIN
    assert out.reason == Reason.SUBRESONANCE_LIBRATES
    assert out.owner == '2J-1-1+0'


@pytest.mark.parametrize('status', [Status.RESONANT, Status.TRANSIENT, Status.NEAR_SEPARATRIX, Status.UNCERTAIN])
def test_member_status_is_copied(status):
    angle = decide.AngleDecision(AngleStatus.CIRCULATION, AngleSubtype.NONE, 'G4')
    member = _member_verdict(status, AngleStatus.LIBRATION)
    out = decide.member_status(angle, _sub_result('2J-1-1+0'), member, Ledger.from_settings(SETTINGS))
    assert out.status == status
    assert out.owner == '2J-1-1+0'


def test_member_status_both_librate_keeps_own_verdict_and_no_member_row():
    angle = decide.AngleDecision(AngleStatus.LIBRATION, AngleSubtype.NONE, 'G2')
    member = _member_verdict(Status.RESONANT, AngleStatus.LIBRATION)
    ledger = Ledger.from_settings(SETTINGS)
    assert decide.member_status(angle, _sub_result('2J-1-1+0'), member, ledger) is None
    assert decide.members_locked(angle, member) == 1
    # without the member's own row (a nodal member) rule C does not apply
    circ = decide.AngleDecision(AngleStatus.CIRCULATION, AngleSubtype.NONE, 'G4')
    assert decide.member_status(circ, _sub_result('2J-1-1+0'), None, ledger) is None
    assert math.isnan(decide.members_locked(circ, None))


# --------------------------------------------------------------------------------------------- L4, variant D


def _force_data(phase_drift_rad, noise=0.0, span=20000.0, step=1.0, seed=3):
    """sigma = pi + 1 rad sin(w0 t), a_dot = -sin(sigma - phi0(t)) + noise, phi0 = pi + 0.3 + drift t / span."""
    t = np.arange(0.0, span + step / 2, step)
    sigma = np.pi + 1.0 * np.sin(W0 * t)
    phi0 = np.pi + 0.3 + phase_drift_rad * t / span
    adot = -np.sin(sigma - phi0)
    if noise:
        adot = adot + noise * np.random.default_rng(seed).standard_normal(t.size)
    return t, sigma, adot


def _windows(t, sigma, adot):
    return response.force_windows(t, sigma, adot, SETTINGS.mmr_force_window_cycles * P0_YR, SETTINGS.mmr_force_window_step)


def test_phase_shifted_clean_force_law_passes_l4():
    t, sigma, adot = _force_data(phase_drift_rad=math.radians(90))
    windows = _windows(t, sigma, adot)
    assert len(windows) >= SETTINGS.mmr_force_min_windows
    assert min(w.r2 for w in windows) > 0.9  # the phase turns ~10 deg inside a 3.5-cycle window
    # the phase really moves: its spread exceeds the old tolerance (30 deg)
    _, _, max_dev = response.phase_spread(windows)
    assert max_dev > math.radians(SETTINGS.mmr_force_phase_tol_deg)
    check, thin, law = response.a_response_check(windows, None, 1.0, SETTINGS)
    assert check == CheckValue.PASS
    assert not thin
    assert law == 'single'


def test_force_law_without_signal_fails_l4_and_mid_r2_is_thin():
    t, sigma, adot = _force_data(0.0, noise=5.0)
    windows = _windows(t, sigma, adot)
    assert max(w.r2 for w in windows) < SETTINGS.mmr_force_r2_fail
    check, _, _ = response.a_response_check(windows, None, 1.0, SETTINGS)
    assert check == CheckValue.FAIL
    # R^2 = s / (s + n): signal variance of -sin(sigma - phi0) over the libration, noise chosen for R^2 ~ 0.55
    t, sigma, adot = _force_data(0.0)
    signal_var = np.var(adot)
    noise = math.sqrt(signal_var * (1 / 0.55 - 1))
    t, sigma, adot = _force_data(0.0, noise=noise)
    windows = _windows(t, sigma, adot)
    half = len(windows) // 2
    medians = [np.median([w.r2 for w in windows[:half]]), np.median([w.r2 for w in windows[half:]])]
    assert all(SETTINGS.mmr_force_r2_fail < m < SETTINGS.mmr_force_r2_full for m in medians)
    check, thin, _ = response.a_response_check(windows, None, 1.0, SETTINGS)
    assert check == CheckValue.PASS
    assert thin


# ------------------------------------------------------------------------- 37087: the kinematic episode, B5


def _kin(modulation=float('nan'), two_omega=float('nan')):
    return blank(Kinematics, modulation_period_yr=modulation, two_omega_period_yr=two_omega)


def _rsp(identity, r2_h1, r2_h2, r2_multi):
    multiplet = MultipletFit(harmonics=(0, 1), coefficients=np.zeros(5), r2=r2_multi, coefficient_coherence=0.0, eps=())
    return blank(Response, identity=identity, force_r2_h1=r2_h1, force_r2_h2=r2_h2, multiplet=multiplet)


def test_secular_period_offset_takes_the_closer_period():
    # 37087: P_cyc 5314 yr against 2 pi / (g - g5) = 5709 yr
    assert decide.secular_period_offset(5314.0, _kin(5709.0)) == pytest.approx(abs(5314 / 5709 - 1))
    assert decide.secular_period_offset(5314.0, _kin(9000.0, 5500.0)) == pytest.approx(abs(5314 / 5500 - 1))
    assert math.isnan(decide.secular_period_offset(5314.0, _kin()))


def test_episode_cycle_is_kinematic_needs_all_three():
    kin = _kin(5709.0)
    bad = _rsp(CheckValue.FAIL, 3e-4, 3e-4, 1e-3)
    angle = decide.AngleDecision(AngleStatus.LIBRATION, AngleSubtype.NONE, 'G2', episode_id=0)
    assert decide.episode_cycle_is_kinematic(kin, bad, 5314.0, SETTINGS)
    assert decide.kinematic_check(angle, kin, bad, 5314.0, SETTINGS) == CheckValue.FAIL
    # the period 4000 yr is 30 % off the secular period
    assert not decide.episode_cycle_is_kinematic(kin, bad, 4000.0, SETTINGS)
    # the identity holds (a Hilda follows the forced varpi but a carries its libration)
    assert not decide.episode_cycle_is_kinematic(kin, _rsp(CheckValue.PASS, 3e-4, 3e-4, 1e-3), 5314.0, SETTINGS)
    # a force law in one half
    assert not decide.episode_cycle_is_kinematic(kin, _rsp(CheckValue.FAIL, 0.9, 3e-4, 1e-3), 5314.0, SETTINGS)
    # a multiplet law
    assert not decide.episode_cycle_is_kinematic(kin, _rsp(CheckValue.FAIL, 3e-4, 3e-4, 0.9), 5314.0, SETTINGS)
    # no secular period known
    assert not decide.episode_cycle_is_kinematic(_kin(), bad, 5314.0, SETTINGS)
    ok = _rsp(CheckValue.PASS, 0.95, 0.95, 0.95)
    assert decide.kinematic_check(angle, kin, ok, 5314.0, SETTINGS) == CheckValue.NOT_APPLICABLE


def test_b5_geometry_route_needs_a_full_turn_overlapping_the_episode():
    assert response.centre_turns_in_episode((1.2, 1000.0, 3000.0), (2500.0, 6000.0))
    assert not response.centre_turns_in_episode((1.2, 1000.0, 3000.0), (3500.0, 6000.0))
    assert not response.centre_turns_in_episode((0.9, 1000.0, 3000.0), (2500.0, 6000.0))
    assert not response.centre_turns_in_episode((float('nan'), 1000.0, 3000.0), (2500.0, 6000.0))


# ---------------------------------------------------------------------------------- geometry: the slow edge


H = 500.0
A = 1.0


def _turning_points(times, values, step=1.0):
    times = np.asarray(times, float)
    values = np.asarray(values, float)
    kinds = np.where(np.arange(times.size) % 2 == 0, 1, -1)
    if values[0] < values[1]:
        kinds = -kinds
    t = np.arange(0.0, times[-1] + 200.0 + step / 2, step)
    tp = geo.TurningPoints(indices=np.searchsorted(t, times), times_yr=times, values_rad=values, kinds=kinds, prominence_rad=0.0175)
    return t, tp


def _episodes(times, values):
    t, tp = _turning_points(times, values)
    return geo.find_episodes(t, tp, (t[0], t[-1]), SETTINGS)


def _drift(n, b):
    times = list(100.0 + H * np.arange(n))
    values = [(A if i % 2 == 0 else -A) + b * i for i in range(n)]
    return times, values


def test_slow_drifting_last_half_cycle_is_trimmed():
    """13 TPs drifting by b = 0.21 per half-cycle: margin 2 - 10 b < 0 (rejected); without the last TP 2 - 9 b > 0.
    The last half-cycle lasts 2.2 x the others and goes 1 rad further: it is cut, the 12 TPs are an episode."""
    b = 0.21
    times, values = _drift(12, b)
    last = (A if 12 % 2 == 0 else -A) + 12 * b
    # with a regular last half-cycle the same run drifts and nothing is cut
    regular = _episodes(times + [times[-1] + H], values + [last])
    assert not any(e.accepted for e in regular)
    assert regular[0].reject_reason == geo.RejectReason.CENTRE_MOVES
    # the slow, drifting last half-cycle
    episodes = _episodes(times + [times[-1] + 2.2 * H], values + [last + 1.0])
    accepted = [e for e in episodes if e.accepted]
    assert len(accepted) == 1
    assert accepted[0].n_tp == 12
    assert accepted[0].t_end_yr == pytest.approx(times[-1])
    edges = [e for e in episodes if e.reject_reason == geo.RejectReason.EDGE]
    assert len(edges) == 1 and edges[0].n_tp == 1
    assert edges[0].t_start_yr == pytest.approx(times[-1] + 2.2 * H)
    ids = [e.episode_id for e in episodes]
    assert ids == list(range(ids[0], ids[0] + len(ids)))


def test_slow_first_half_cycle_is_trimmed_symmetrically():
    b = 0.21
    times, values = _drift(12, b)
    first_time = times[0] - 2.2 * H
    first_value = -A - b - 1.0  # a minimum before the first maximum, 1 rad further down
    episodes = _episodes([first_time] + times, [first_value] + values)
    accepted = [e for e in episodes if e.accepted]
    assert len(accepted) == 1 and accepted[0].n_tp == 12
    assert accepted[0].t_start_yr == pytest.approx(times[0])
    assert episodes[0].reject_reason == geo.RejectReason.EDGE


def test_uniform_drift_is_not_accepted_on_a_short_piece():
    """b = 0.35: 13 TPs (2 - 10 b < 0) and 12 TPs (2 - 9 b < 0) drift, 8 TPs (2 - 5 b > 0) would pass the band test:
    one slow half-cycle may be cut, never more."""
    b = 0.35
    times, values = _drift(13, b)
    short = _episodes(times[:8], values[:8])
    assert any(e.accepted for e in short)  # the danger: a short piece of a drift looks stationary
    times[-1] = times[-2] + 2.2 * H  # the last half-cycle slow
    episodes = _episodes(times, values)
    assert not any(e.accepted for e in episodes)


# --------------------------------------------------------------------------------- planets: ripple, catalogue


GI = GreatInequality(period_yr=880.0, dlambda_jupiter_rad=math.radians(0.34), dlambda_saturn_rad=math.radians(0.83), measured=False)


@pytest.mark.parametrize(
    'key, expected_deg',
    [
        ('3J+1S-1', abs(3 * 0.34 - 1 * 0.83)),  # same signs: the antiphase waves cancel (0.19 deg)
        ('3J-1S-1', 3 * 0.34 + 1 * 0.83),  # opposite signs: they add (1.85 deg)
        ('5J+6S-1', abs(5 * 0.34 - 6 * 0.83)),  # 3.28 deg, not 6.68
        ('4J-2S-1', 4 * 0.34 + 2 * 0.83),
    ],
)
def test_great_inequality_ripple_by_hand(key, expected_deg):
    spec = ResonanceSpec.from_key(key)
    # a cutoff of 10 yr leaves the 880-yr wave untouched: G = 1
    ripple = predicted_ripple(spec, GI, 10.0, 4)
    assert math.degrees(ripple) == pytest.approx(expected_deg, rel=1e-3)


def test_ripple_ratio_is_independent_of_the_filter_gain():
    same = predicted_ripple(ResonanceSpec.from_key('3J+1S-1'), GI, 600.0, 4)
    opposite = predicted_ripple(ResonanceSpec.from_key('3J-1S-1'), GI, 600.0, 4)
    assert same / opposite == pytest.approx((3 * 0.34 - 0.83) / (3 * 0.34 + 0.83), rel=1e-6)
    assert predicted_ripple(ResonanceSpec.from_key('2M-1'), None, 10.0, 4) == 0.0


def test_catalogue_phase_error_bound_by_hand():
    spec = ResonanceSpec.from_key('5J-2S-2')
    # T = 1 kyr: the rate spread decides: 5 x 1.5e-4 x 1000 + 2 x 3.7e-4 x 1000 = 0.75 + 0.74 deg
    assert math.degrees(owner.catalogue_phase_error(spec, ('Jupiter', 'Saturn'), 1000.0)) == pytest.approx(1.49)
    # T = 100 kyr: the slow wander + offset decides: 5 (2 x 1.91 + 0.41) + 2 (2 x 4.69 + 0.85) = 21.15 + 20.46 deg
    assert math.degrees(owner.catalogue_phase_error(spec, ('Jupiter', 'Saturn'), 1.0e5)) == pytest.approx(41.61)
    # only the catalogued planets count
    assert math.degrees(owner.catalogue_phase_error(spec, ('Saturn',), 1000.0)) == pytest.approx(0.74)
    assert owner.catalogue_phase_error(spec, (), 1000.0) == 0.0
    # 9 lambda_S of 5M-9S-4 (2003 UG22) over 18.9 kyr: D = 9 x 3.7e-4 x 18900 deg = 1.10 rad
    assert owner.catalogue_phase_error(ResonanceSpec.from_key('5M-9S-4'), ('Saturn',), 18900.0) == pytest.approx(1.1, abs=0.01)


def test_sinc_half_slope_max():
    """Delta C per unit D: the steepest slope of |sinc(D / 2)| = |sin(x) / x|, x = D / 2, is max |f'(x)| / 2."""
    x = np.linspace(1e-6, np.pi, 200001)
    slope = np.abs((x * np.cos(x) - np.sin(x)) / x**2) / 2
    assert slope.max() == pytest.approx(owner.SINC_HALF_SLOPE_MAX, abs=1e-4)


def test_catalogue_longitudes_carry_lambda_with_the_table_rate():
    out = catalogue_longitudes({'Earth': 1.0, 'Jupiter': 2.0, 'Mercury': 0.5}, 0.0, 0.5)
    assert 'Mercury' not in out  # never catalogued
    earth = (1.0 + math.radians(360.000505 * 0.5)) % (2 * np.pi)
    jupiter = (2.0 + math.radians(30.349646 * 0.5)) % (2 * np.pi)
    assert out['Earth'] == pytest.approx(earth, abs=1e-9)
    assert out['Jupiter'] == pytest.approx(jupiter, abs=1e-9)
