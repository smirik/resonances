"""Independent tests of the W9 changes (planets, strength well, decide, slow edge half-cycles).

Written from the W9 API notes, the public signatures and docs/libration-analysis.md only. Every expected value is
derived here by hand or from a synthetic model with a known answer:

- strength: the averaged <D>(sigma) of a Plutino (3:2 with Neptune; minimum at 180 deg, maximum at 0 deg where the
  conjunction falls at perihelion) and of a Twotino (exterior 1:2: two asymmetric wells, inner maximum at 180 deg,
  global maximum at 0 deg; Beauge 1994, Gallardo 2006 sect. 3.1); swings inside a well, over the top, horseshoes;
- decide: a synthetic pendulum record (sigma_dot = 2 n_J - n(a), a_dot = -K sin(sigma - pi)) that circulates and is
  captured after a jump of a; rule C (`member_status`) on hand-built verdicts; L4 on force windows whose phase phi0
  wanders but whose R^2 is high;
- geometry: turning points with a slow, drifting last half-cycle (trimmed) and a uniform drift (never accepted);
- planets: the great-inequality ripple |k_J dl_J - k_S dl_S| G(P) by hand, the catalogue phase-error bound
  D = sum |k_j| min(eps T, 2 A + delta T), the sinc slope 0.2181 and the catalogue line lambda_j0 + n_j t.

Fast: no N-body integration (the synthetic records take < 1 s each).
"""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from resonances.data import const
from resonances.resonance.analysis import geometry as geo
from resonances.resonance.analysis.ledger import Ledger
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
from resonances.resonance.analysis.mmr import averaged_potential as ap
from resonances.resonance.analysis.mmr import context as ctx
from resonances.resonance.analysis.mmr import owner
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.coorbital import Orbit
from resonances.resonance.analysis.mmr.decide import AngleDecision, member_status, members_locked
from resonances.resonance.analysis.mmr.models import (
    ForceWindow,
    GreatInequality,
    MMRSeries,
    Regime,
    ResonanceSpec,
    SubresonanceCandidate,
    SubresonanceResult,
)
from resonances.resonance.analysis.mmr.preprocess import predicted_ripple
from resonances.resonance.analysis.mmr.response import a_response_check
from resonances.resonance.analysis.mmr.strength import observed_swing, quadrature

SETTINGS = AnalysisSettings()
DEG = math.pi / 180


# ============================================================================================ strength: the well


@pytest.fixture(scope='module')
def neptune():
    return ap.Ellipse(a=30.07, orbit=Orbit(e=0.0, inc=0.0, omega=0.0, Omega=0.0))


@pytest.fixture(scope='module')
def plutino(neptune):
    """sigma = 3 lambda - 2 lambda_N - varpi = 3 M + 2 varpi - 2 lambda_N: k = 3, k_P = -2, phase 2 varpi = 0.
    q = 29.6 au < a_N: at sigma = 0 the conjunction falls at perihelion (the top of <D>), at 180 deg at aphelion."""
    body = ap.Ellipse(a=39.45, orbit=Orbit(e=0.25, inc=0.1, omega=0.0, Omega=0.0))
    return ap.centre_curvature(body, neptune, 3, -2, 0.0, math.pi - 0.4, quadrature(SETTINGS))


@pytest.fixture(scope='module')
def twotino(neptune):
    """Exterior 1:2, e = 0.3: asymmetric wells near +-70 deg, an inner maximum at 180 deg, the global one at 0."""
    body = ap.Ellipse(a=47.8, orbit=Orbit(e=0.3, inc=0.1, omega=0.0, Omega=0.0))
    return ap.centre_curvature(body, neptune, 2, -1, 0.0, 80 * DEG, quadrature(SETTINGS))


def test_plutino_well_has_its_minimum_at_180_and_global_max_at_0(plutino):
    assert plutino.measured
    assert abs(ap.circular_distance(plutino.centre_rad, math.pi)) < 2 * DEG
    assert abs(ap.circular_distance(plutino.global_max_rad, 0.0)) < 6 * DEG


def test_twotino_asymmetric_well_and_global_max(twotino):
    # Beauge 1994 / Gallardo 2006: asymmetric centres at ~60-80 deg for e ~ 0.3; the global top stays at 0.
    assert twotino.measured
    assert 55 * DEG < twotino.centre_rad < 85 * DEG
    assert abs(ap.circular_distance(twotino.global_max_rad, 0.0)) < 6 * DEG


@pytest.mark.parametrize(
    'lo_deg, hi_deg, inside',
    [
        (120, 240, True),  # a libration about 180 deg
        (179, 181, True),  # tiny swing
        (100, 170, False),  # does not contain the centre
        (120, 365, False),  # passes the top at 360 = 0 deg
        (-60, 200, False),  # passes the top at 0 deg on the other side
        (120 + 5 * 360, 240 + 5 * 360, True),  # the same swing five turns away (unwrapped axis)
        (120 - 3 * 360, 240 - 3 * 360, True),
    ],
)
def test_swing_in_plutino_well(plutino, lo_deg, hi_deg, inside):
    assert ap.swing_in_well(plutino.centre_rad, plutino.global_max_rad, lo_deg * DEG, hi_deg * DEG) is inside


@pytest.mark.parametrize(
    'lo_deg, hi_deg, inside',
    [
        (50, 95, True),  # tadpole in the leading asymmetric well
        (40, 300, True),  # horseshoe about both wells: crosses only the inner maximum at 180 deg
        (-30, 95, False),  # crosses the global maximum at 0 deg
        (100, 170, False),  # misses the occupied centre
    ],
)
def test_swing_in_twotino_well(twotino, lo_deg, hi_deg, inside):
    assert ap.swing_in_well(twotino.centre_rad, twotino.global_max_rad, lo_deg * DEG, hi_deg * DEG) is inside


def test_swing_in_well_with_synthetic_potential():
    # <D> = -cos(s - c0) + 0.2 cos(2 (s - c0)), c0 = 1 rad: D''(c0) = 1 - 0.8 > 0 (the minimum), D''(c0 + pi) =
    # -1 - 0.8 < 0 (the maximum, D = 1.2)
    c0 = 1.0
    s = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
    d = -np.cos(s - c0) + 0.2 * np.cos(2 * (s - c0))
    centre, top = s[np.argmin(d)], s[np.argmax(d)]
    assert abs(centre - c0) < 1e-2 and abs(ap.circular_distance(top, c0 + np.pi)) < 1e-2
    assert ap.swing_in_well(centre, top, c0 - 1.0, c0 + 2.0)
    assert not ap.swing_in_well(centre, top, c0 - 1.0, c0 + 3.3)  # over the top at c0 + pi
    assert not ap.swing_in_well(centre, top, c0 + 0.2, c0 + 2.0)  # centre not inside
    assert not ap.swing_in_well(float('nan'), top, c0 - 1.0, c0 + 1.0)
    assert not ap.swing_in_well(centre, float('nan'), c0 - 1.0, c0 + 1.0)


# ============================================================================================ geometry


def _tp(times, values, first_kind=1):
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    kinds = np.where(np.arange(len(values)) % 2 == 0, first_kind, -first_kind)
    return geo.TurningPoints(indices=np.arange(len(times)), times_yr=times, values_rad=values, kinds=kinds, prominence_rad=0.05)


def _episodes(times, values, pad=300.0):
    t = np.arange(times[0] - pad, times[-1] + pad + 1.0, 1.0)
    return geo.find_episodes(t, _tp(times, values), (t[0], t[-1]), SETTINGS)


HALF = 600.0  # median half-period [yr]
DRIFT = 0.2  # rad per TP: 12 TPs stay stationary, a 13th TP on the same trend does not (measured below)


def _drifting(n, drift=DRIFT, centre=np.pi):
    return [centre + (1.0 if i % 2 == 0 else -1.0) + drift * i for i in range(n)]


def test_drift_threshold_of_the_synthetic_run():
    """The premise of the next tests: 12 TPs on this drift are one stationary episode, 13 are not."""
    times12 = list(np.arange(12) * HALF)
    eps12 = _episodes(times12, _drifting(12))
    assert [e.accepted for e in eps12] == [True]
    times13 = list(np.arange(13) * HALF)
    eps13 = _episodes(times13, _drifting(13))
    assert not any(e.accepted for e in eps13)  # uniform drift, equal half-periods: nothing to trim


def test_slow_drifting_last_half_cycle_is_trimmed():
    # the last half-cycle takes 1400 yr = 2.33 x the 600-yr median half-period, and carries the centre away;
    # its full period (600 + 1400 = 2000 yr) is only 1.67 x the 1200-yr median period: `_split_regular` misses it
    times = list(np.arange(12) * HALF) + [11 * HALF + 1400.0]
    values = _drifting(13)
    eps = _episodes(times, values)
    accepted = [e for e in eps if e.accepted]
    assert len(accepted) == 1
    core = accepted[0]
    assert core.n_tp == 12
    assert core.tp_times_yr[0] == 0.0 and core.tp_times_yr[-1] == 11 * HALF
    edges = [e for e in eps if not e.accepted]
    assert len(edges) == 1
    assert edges[0].reject_reason == geo.RejectReason.EDGE
    assert list(edges[0].tp_times_yr) == [times[-1]]
    ids = [e.episode_id for e in eps]
    assert ids == sorted(ids) and len(set(ids)) == len(ids) and ids == list(range(ids[0], ids[0] + len(ids)))


def test_slow_drifting_first_half_cycle_is_trimmed():
    # the mirror image: a slow first half-cycle
    times = [-1400.0] + list(np.arange(12) * HALF)
    values = [np.pi - 1.0 - DRIFT] + _drifting(12)
    # alternation: the first TP is a minimum, then the run of _drifting (max first)
    tp = _tp(times, values, first_kind=-1)
    t = np.arange(times[0] - 300.0, times[-1] + 301.0, 1.0)
    eps = geo.find_episodes(t, tp, (t[0], t[-1]), SETTINGS)
    accepted = [e for e in eps if e.accepted]
    assert len(accepted) == 1 and accepted[0].n_tp == 12 and accepted[0].tp_times_yr[0] == 0.0
    assert [e.reject_reason for e in eps if not e.accepted] == [geo.RejectReason.EDGE]


def test_half_cycle_below_the_ratio_is_not_trimmed():
    # 1080 yr = 1.8 x the median half-period: not slow, so the drifting run is not cut
    times = list(np.arange(12) * HALF) + [11 * HALF + 1080.0]
    eps = _episodes(times, _drifting(13))
    assert not any(e.accepted for e in eps)
    assert geo.RejectReason.EDGE not in [e.reject_reason for e in eps]


def test_uniform_drift_is_not_accepted_on_a_short_piece():
    # a steeper drift: without its slow last half-cycle the core still drifts. Shorter pieces of it would pass
    # (4 TPs of any drift below the band margin), but at most one half-cycle is dropped per edge.
    drift = 0.35
    short = _episodes(list(np.arange(4) * HALF), _drifting(4, drift))
    assert [e.accepted for e in short] == [True]  # premise: a short piece of this drift is stationary
    times = list(np.arange(12) * HALF) + [11 * HALF + 1400.0]
    eps = _episodes(times, _drifting(13, drift))
    assert not any(e.accepted for e in eps)


def test_observed_swing_of_an_episode_and_the_well(plutino):
    # an episode about pi with half-amplitude 1 rad, unwrapped five turns up
    off = 10 * np.pi
    times = list(np.arange(9) * HALF)
    values = [off + np.pi + (1.0 if i % 2 == 0 else -1.0) for i in range(9)]
    (episode,) = [e for e in _episodes(times, values) if e.accepted]
    lo, hi = observed_swing(episode, 3.5 * HALF)  # the half-cycle 3 -> 4
    assert lo == pytest.approx(off + np.pi - 1.0) and hi == pytest.approx(off + np.pi + 1.0)
    assert ap.swing_in_well(plutino.centre_rad, plutino.global_max_rad, lo, hi)


def test_observed_swing_needs_two_turning_points():
    times = list(np.arange(12) * HALF) + [11 * HALF + 1400.0]
    eps = _episodes(times, _drifting(13))
    (edge,) = [e for e in eps if e.reject_reason == geo.RejectReason.EDGE]
    lo, hi = observed_swing(edge, times[-1])
    assert math.isnan(lo) and math.isnan(hi)


# ============================================================================================ decide: G1l (capture)

N_J = math.radians(const.PLANET_N_DEG_PER_GAUSS_YR['Jupiter'])
A_STAR = (2 * math.pi / (2 * N_J)) ** (2.0 / 3.0)  # n(a*) = 2 n_J for 2J-1
K_FORCE = 5.1e-4  # au/yr^2: omega0 = sqrt(1.5 n/a K) ~ 0.0157 rad/yr, P ~ 400 yr, half-width ~ 0.065 au


def _n(a):
    return 2 * np.pi * a**-1.5


def _rhs(centre):
    def rhs(_t, y):
        return [2 * N_J - _n(y[1]), -K_FORCE * np.sin(y[0] - centre)]

    return rhs


def _integrate(sigma0, a0, t0, t1, step, centre):
    t = np.arange(t0, t1, step)
    sol = solve_ivp(_rhs(centre), (t[0], t[-1]), [sigma0, a0], t_eval=t, rtol=1e-10, atol=1e-12, method='DOP853')
    return sol.t, sol.y[0], sol.y[1]


def _series(t, sigma, a, key='2J-1', name='synthetic'):
    n = len(t)
    z = np.zeros(n)
    return MMRSeries(
        t_yr=t,
        a=a,
        e=z + 0.15,
        inc=z + 0.05,
        Omega=z + 1.0,
        omega=z + 0.5,
        varpi=z + 1.5,
        angle_wrapped=np.mod(sigma, 2 * np.pi),
        resonance=ResonanceSpec.from_key(key),
        regime=Regime(step_yr=float(t[1] - t[0]), span_yr=float(t[-1] - t[0]), n_rows=n),
        name=name,
    )


def _capture(da_before, da_after, t_jump=9000.0, span=20000.0, step=1.0):
    """Circulation with a = a* + da_before until sigma passes the centre pi after t_jump, then a jumps to
    a* + da_after (a libration of ~55 deg about pi) to the end of the record."""
    t1, s1, a1 = _integrate(np.pi, A_STAR + da_before, 0.0, t_jump + 3000.0, step, np.pi)
    wrapped = (s1 - np.pi + np.pi) % (2 * np.pi) - np.pi
    i = int(np.where((t1 >= t_jump) & (np.abs(wrapped) < 0.05))[0][0])
    t1, s1, a1 = t1[: i + 1], s1[: i + 1], a1[: i + 1]
    t2, s2, a2 = _integrate(s1[-1], A_STAR + da_after, t1[-1] + step, span + step / 2, step, np.pi)
    return _series(np.concatenate([t1, t2]), np.concatenate([s1, s2]), np.concatenate([a1, a2])), float(t1[-1])


CAPTURES = [
    pytest.param(0.25, 0.035, id='prograde-circulation'),
    pytest.param(-0.25, 0.035, id='retrograde-circulation'),
    pytest.param(0.30, 0.03, id='fast-circulation'),
]


@pytest.mark.slow
@pytest.mark.parametrize('da_before, da_after', CAPTURES)
def test_capture_after_jump_is_transient_with_final_libration(da_before, da_after):
    series, t_jump = _capture(da_before, da_after)
    verdict = analyse_mmr(series, SETTINGS).verdict
    assert verdict.angle_status == AngleStatus.EPISODIC
    assert verdict.status == Status.TRANSIENT
    assert verdict.final_regime == AngleStatus.LIBRATION
    # the libration starts at the jump (within the jump zone and one libration period of ~430 yr)
    assert t_jump <= verdict.final_regime_start_yr <= t_jump + 450.0


@pytest.mark.slow
@pytest.mark.parametrize(
    'da_before, da_after',
    [
        pytest.param(0.25, 0.035, id='prograde-circulation'),
        pytest.param(-0.25, 0.035, id='retrograde-circulation'),
    ],
)
def test_capture_after_circulation_is_not_recurrent(da_before, da_after):
    series, _ = _capture(da_before, da_after)
    verdict = analyse_mmr(series, SETTINGS).verdict
    assert verdict.angle_subtype != AngleSubtype.RECURRENT  # one librating episode only


def test_libration_through_the_record_stays_resonant():
    # the control: one libration (amplitude ~36 deg) from the start to the end is a libration, not a capture. About 0,
    # where Jupiter's directly averaged potential has its minimum for 2J-1 (352 deg): a libration about pi would lie
    # outside that well, and with L4 and R0 not measured the failed strength check makes it uncertain (W18)
    t, s, a = _integrate(0.0, A_STAR + 0.02, 0.0, 20000.5, 1.0, 0.0)
    verdict = analyse_mmr(_series(t, s, a), SETTINGS).verdict
    assert verdict.angle_status == AngleStatus.LIBRATION
    assert verdict.status == Status.RESONANT


# ============================================================================================ decide: L4 without phi0


def _windows(phi0_deg, r2, span=20000.0):
    n = len(phi0_deg)
    edges = np.linspace(0.0, span, n + 1)
    return [
        ForceWindow(
            t_start_yr=float(edges[i]),
            t_end_yr=float(edges[i + 1]),
            r2=float(r2[i] if np.ndim(r2) else r2),
            amplitude_au_yr=1e-4,
            phi0_rad=math.radians(phi0_deg[i]),
            offset_au_yr=0.0,
        )
        for i in range(n)
    ]


def test_l4_passes_a_clean_law_whose_phase_wanders():
    # phi0 runs over 0..150 deg (far beyond the 30 deg tolerance column), R^2 = 0.95 in every window
    windows = _windows(np.linspace(0.0, 150.0, 8), 0.95)
    check, thin, _law = a_response_check(windows, None, math.radians(80.0), SETTINGS)
    assert check == CheckValue.PASS
    assert not thin


def test_l4_thin_and_fail_by_r2_alone():
    phases = np.linspace(0.0, 150.0, 8)
    mid = 0.5 * (SETTINGS.mmr_force_r2_fail + SETTINGS.mmr_force_r2_full)
    check, thin, _ = a_response_check(_windows(phases, mid), None, math.radians(80.0), SETTINGS)
    assert check == CheckValue.PASS and thin
    check, _, _ = a_response_check(_windows(phases, 0.5 * SETTINGS.mmr_force_r2_fail), None, math.radians(80.0), SETTINGS)
    assert check == CheckValue.FAIL
    # one bad half: R^2 is the smaller of the half medians
    r2 = [0.95] * 4 + [0.2] * 4
    check, _, _ = a_response_check(_windows(phases, r2), None, math.radians(80.0), SETTINGS)
    assert check == CheckValue.FAIL


def test_l4_not_measured_below_the_amplitude_or_windows():
    windows = _windows(np.zeros(8), 0.95)
    small = math.radians(0.5 * SETTINGS.mmr_a_response_min_amp_deg)
    assert a_response_check(windows, None, small, SETTINGS)[0] == CheckValue.NOT_MEASURED
    few = _windows(np.zeros(max(SETTINGS.mmr_force_min_windows - 1, 0)), 0.95)
    assert a_response_check(few, None, math.radians(80.0), SETTINGS)[0] in (CheckValue.NOT_MEASURED, CheckValue.NOT_APPLICABLE)


def test_shifted_centre_pendulum_passes_l4_end_to_end():
    # a synthetic pendulum about centre 1.0 rad (not 0 or pi): the force law is clean, L4 must pass
    centre = 1.0
    t, s, a = _integrate(centre, A_STAR + 0.04, 0.0, 20000.5, 1.0, centre)
    verdict = analyse_mmr(_series(t, s, a), SETTINGS).verdict
    assert verdict.checks['check_mmr_a_response'] == CheckValue.PASS
    assert verdict.angle_status == AngleStatus.LIBRATION


# ============================================================================================ decide: rule C


def _member_result(key='9M-16+2+5', accepted=True):
    candidate = SubresonanceCandidate(
        spec=ResonanceSpec.from_key(key),
        j=2,
        nodal=False,
        centre_range_rad=0.1,
        centre_rate_rad_yr=0.0,
        centre_max_pass_rad=0.1,
        centre_mean_rad=0.0,
        phi0_coherence=0.95,
        coefficient_rel=1.0,
        composite_rms_rad=0.1,
        period_corr=0.9,
        period_ratio=1.0,
        rank=1,
        accepted=accepted,
        reason='',
    )
    return SubresonanceResult(
        triggers=(),
        support=(),
        candidates=[candidate],
        accepted=candidate if accepted else None,
        check=CheckValue.FAIL if accepted else CheckValue.PASS,
        composite=False,
        undetermined=False,
        tie_by_order=False,
        theory_measured=True,
        sigma0_phi0_coherence=0.5,
        composite_rms_rad=0.1,
        single_rms_rad=0.1,
        binomial_mode=2,
        drift_pred_rad_yr=0.0,
        drift_meas_rad_yr=0.0,
        plane_offset_rad=0.0,
    )


def _member(status, angle=AngleStatus.LIBRATION):
    return Verdict(resonance_type=ResonanceType.MMR, status=status, angle_status=angle)


CIRCULATING = AngleDecision(AngleStatus.CIRCULATION, AngleSubtype.NONE, 'G4')
DRIFTING = AngleDecision(AngleStatus.DRIFT, AngleSubtype.NONE, 'G5')


@pytest.mark.parametrize(
    'member, expected',
    [
        (Status.RESONANT, Status.RESONANT),
        (Status.TRANSIENT, Status.TRANSIENT),
        (Status.NEAR_SEPARATRIX, Status.NEAR_SEPARATRIX),
        (Status.NON_RESONANT, Status.NON_RESONANT),
        (Status.UNCERTAIN, Status.UNCERTAIN),
        (Status.INVALID, Status.UNCERTAIN),
    ],
)
@pytest.mark.parametrize('angle', [CIRCULATING, DRIFTING], ids=['circulation', 'drift'])
def test_catalogue_row_takes_the_member_status(angle, member, expected):
    member_angle = AngleStatus.LIBRATION if member == Status.RESONANT else AngleStatus.EPISODIC
    decision = member_status(angle, _member_result(), _member(member, member_angle), Ledger.from_settings(SETTINGS))
    assert decision is not None
    assert decision.status == expected
    assert decision.reason == Reason.SUBRESONANCE_LIBRATES
    assert decision.origin == LibrationOrigin.SUBRESONANCE
    assert decision.owner == '9M-16+2+5'
    assert decision.rule == '1'


def test_both_members_librating_keep_their_own_verdicts():
    librating = AngleDecision(AngleStatus.LIBRATION, AngleSubtype.STATIONARY, 'G2', episode_id=0)
    member = _member(Status.RESONANT, AngleStatus.LIBRATION)
    assert member_status(librating, _member_result(), member, Ledger.from_settings(SETTINGS)) is None
    assert members_locked(librating, member) == 1
    # the catalogue angle librates over the record, the member's row only in an episode: 0
    assert members_locked(librating, _member(Status.TRANSIENT, AngleStatus.EPISODIC)) == 0
    # the catalogue angle does not librate: nan (the docstring's "0 when only the member's does" contradicts this)
    assert math.isnan(members_locked(CIRCULATING, member))


def test_rule_c_needs_an_accepted_member_and_its_row():
    ledger = Ledger.from_settings(SETTINGS)
    assert member_status(CIRCULATING, _member_result(), None, ledger) is None
    assert member_status(CIRCULATING, _member_result(accepted=False), _member(Status.RESONANT), ledger) is None
    assert member_status(CIRCULATING, None, _member(Status.RESONANT), ledger) is None
    assert math.isnan(members_locked(CIRCULATING, None))


# ============================================================================================ planets: ripple (C11)


GI = GreatInequality(
    period_yr=937.0,
    dlambda_jupiter_rad=math.radians(0.34),
    dlambda_saturn_rad=math.radians(0.83),
    measured=False,
)


def _analog_gain(period, cutoff, order=2):
    return 1.0 / (1.0 + (cutoff / period) ** (2 * order))


def _digital_gain(period, cutoff, step, order=2):
    return 1.0 / (1.0 + (math.tan(math.pi * step / period) / math.tan(math.pi * step / cutoff)) ** (2 * order))


@pytest.mark.parametrize(
    'key, unsigned_deg',
    [
        ('5J+6S-1', abs(5 * 0.34 - 6 * 0.83)),  # same sign: the waves cancel partly, 3.28 deg (not 6.68)
        ('5J-2S-2', abs(5 * 0.34 + 2 * 0.83)),  # opposite sign: they add, 3.36 deg
        ('2J-1', 2 * 0.34),
        ('1S-2', 0.83),
        ('9M-16+0+7', 0.0),  # neither Jupiter nor Saturn
    ],
)
def test_great_inequality_ripple_by_hand(key, unsigned_deg):
    spec = ResonanceSpec.from_key(key)
    # P_c = P_GI: the analog gain is exactly 1/2
    assert predicted_ripple(spec, GI, 937.0, 2) == pytest.approx(math.radians(unsigned_deg) * 0.5, abs=1e-12)
    # P_c = P_GI / 2: 1 / (1 + 1/16) = 16/17
    assert predicted_ripple(spec, GI, 468.5, 2) == pytest.approx(math.radians(unsigned_deg) * 16 / 17, abs=1e-12)
    expected = math.radians(unsigned_deg) * _digital_gain(937.0, 468.5, 10.0)
    assert predicted_ripple(spec, GI, 468.5, 2, step_yr=10.0) == pytest.approx(expected, rel=1e-9, abs=1e-15)


def test_ripple_same_sign_is_below_the_unsigned_sum():
    same = predicted_ripple(ResonanceSpec.from_key('5J+6S-1'), GI, 937.0, 2)
    unsigned = math.radians(5 * 0.34 + 6 * 0.83) * _analog_gain(937.0, 937.0)
    assert same < 0.5 * unsigned  # 3.28 deg against 6.68 deg


# ============================================================================================ planets: catalogue (C)


def _bound_deg(name, span):
    eps = const.PLANET_N_SPREAD_DEG_PER_GAUSS_YR[name]
    delta = const.PLANET_N_OFFSET_DEG_PER_GAUSS_YR[name]
    slow = const.PLANET_LAMBDA_SLOW_MAX_DEG[name]
    return min(eps * span, 2 * slow + delta * span)


def test_catalogue_error_bound_by_hand():
    # Earth, 20 kyr: eps T = 2.4e-5 x 2e4 = 0.48 deg < 2 x 0.37 + 1.1e-5 x 2e4 = 0.96 deg -> 5 x 0.48 = 2.40 deg
    spec = ResonanceSpec.from_key('5E-16+0+11')
    assert owner.catalogue_phase_error(spec, ('Earth',), 20000.0) == pytest.approx(math.radians(2.40), rel=1e-6)
    # 200 kyr: eps T = 4.8 deg > 0.74 + 2.2 = 2.94 deg -> 5 x 2.94 = 14.7 deg (the slow wobble bound takes over)
    assert owner.catalogue_phase_error(spec, ('Earth',), 200000.0) == pytest.approx(math.radians(14.7), rel=1e-6)
    # 2003 UG22 5M-9S-4 with Saturn catalogued: 9 x min(7.4, 9.38 + 0.17) = 66.6 deg = 1.16 rad (docs: 1.1 rad)
    ug22 = ResonanceSpec.from_key('5M-9S-4')
    assert owner.catalogue_phase_error(ug22, ('Saturn',), 20000.0) == pytest.approx(math.radians(66.6), rel=1e-6)
    # Mars too: + 5 x min(0.74, 1.44 + 0.122) = 3.7 deg
    assert owner.catalogue_phase_error(ug22, ('Mars', 'Saturn'), 20000.0) == pytest.approx(math.radians(70.3), rel=1e-6)
    assert owner.catalogue_phase_error(ug22, (), 20000.0) == 0.0


@pytest.mark.parametrize('span', [100.0, 5000.0, 20000.0, 1e5, 1e6])
@pytest.mark.parametrize('key, catalogued', [('5M-9S-4', ('Mars', 'Saturn')), ('3J-2S-1', ('Jupiter', 'Saturn')), ('1E-1', ('Earth',))])
def test_catalogue_error_bound_formula(key, catalogued, span):
    spec = ResonanceSpec.from_key(key)
    coefficients = dict(zip(spec.planets, spec.planet_coefficients))
    expected = sum(abs(coefficients[name]) * _bound_deg(name, span) for name in catalogued)
    assert owner.catalogue_phase_error(spec, catalogued, span) == pytest.approx(math.radians(expected), rel=1e-9)


def test_sinc_slope_constant_bounds_the_coherence_loss():
    # phases drifting linearly over D [rad] across window centres spread uniformly: coherence |sinc(D/2)|
    d = np.linspace(1e-6, 40.0, 400001)
    sinc = np.sin(d / 2) / (d / 2)
    slope = np.abs(np.gradient(sinc, d))
    assert slope.max() <= owner.SINC_HALF_SLOPE_MAX  # a Lipschitz bound of sinc(D/2): 0.21809 at D = 4.16
    assert owner.SINC_HALF_SLOPE_MAX - slope.max() < 1e-4
    assert np.all(1 - np.abs(sinc) <= owner.SINC_HALF_SLOPE_MAX * d + 1e-12)
    # the coherence of unit phasors drifting over D, measured, and the W9 numbers at 22, 46 and 128 deg
    for deg, quoted in [(22.0, 0.9939), (46.0, 0.9734), (128.0, 0.8046)]:
        phases = np.linspace(0.0, math.radians(deg), 20001)
        coherence = abs(np.mean(np.exp(1j * phases)))
        assert coherence == pytest.approx(quoted, abs=1e-4)


def test_catalogue_planets_exclude_mercury():
    assert 'Mercury' not in ctx.CATALOGUE_PLANETS
    assert set(ctx.CATALOGUE_PLANETS) == {'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'}


def test_catalogue_longitudes_carry_lambda_with_the_table_rate():
    out = ctx.catalogue_longitudes({'Earth': 0.0, 'Jupiter': 0.5, 'Mercury': 1.0}, 0.0, 1000.0)
    assert 'Mercury' not in out
    # Earth: 360.000505 deg/yr x 1000 yr = 360000.505 deg = 0.505 deg mod 360
    assert out['Earth'] == pytest.approx(math.radians(0.505), abs=1e-9)
    # Jupiter: 0.5 rad + 30349.646 deg = 0.5 rad + 109.646 deg mod 360
    assert out['Jupiter'] == pytest.approx((0.5 + math.radians(109.646)) % (2 * math.pi), abs=1e-9)
    back = ctx.catalogue_longitudes({'Earth': 0.0}, 1000.0, 0.0)
    assert back['Earth'] == pytest.approx(math.radians(360 - 0.505), abs=1e-9)


def _catalogue_series(catalogue):
    t = np.arange(0.0, 200.0, 0.5)
    n = len(t)
    z = np.zeros(n)
    return MMRSeries(
        t_yr=t,
        a=z + 1.0,
        e=z + 0.1,
        inc=z + 0.1,
        Omega=z,
        omega=z,
        varpi=z,
        angle_wrapped=z,
        resonance=ResonanceSpec.from_key('5E-16+0+11'),
        regime=Regime(step_yr=0.5, span_yr=200.0, n_rows=n),
        catalogue_longitudes=catalogue,
    )


def test_catalogue_line_of_a_series():
    series = _catalogue_series({'Earth': 1.0})
    series.validate()
    assert owner.catalogue_planets(series) == ('Earth',)
    line = owner.catalogue_line(series, 'Earth')
    n_earth = math.radians(const.PLANET_N_DEG_PER_GAUSS_YR['Earth'])
    expected = 1.0 + n_earth * (series.t_yr - series.t_yr[0])
    assert np.allclose(np.angle(np.exp(1j * (line - expected))), 0.0, atol=1e-9)
    lam = owner.planet_longitude(series, 'Earth')
    assert np.allclose(np.angle(np.exp(1j * (lam - expected))), 0.0, atol=1e-9)


@pytest.mark.parametrize('bad', [{'Pluto': 1.0}, {'Earth': float('nan')}, {'Earth': float('inf')}])
def test_catalogue_longitudes_are_validated(bad):
    with pytest.raises(ValueError):
        _catalogue_series(bad).validate()
