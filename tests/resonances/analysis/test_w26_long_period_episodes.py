"""W26: one libration stays one episode where its centre moves fast, and an episodic omega whose episodes are the
forced vector's is kinematic.

1. The turning-point prominence of a secular/ZLK angle (`preprocess._set_prominence`) no longer counts the libration
   itself: the two filters at P_c and P_min split softly, and a libration of period P >= P_min leaves
   (G(P; P_c) - G(P; P_min)) of its swing in sigma_f - lowpass(sigma, P_min) (G = 1 / (1 + (P_c/P)^4), order 2,
   forward-backward). 591986 over 4 Myr (P_c 1874 yr, P_min 3748 yr, a 6.1-kyr libration of 16 deg) had r = 4.35 deg
   and p_TP = 17.4 deg, and was cut into three episodes where the centre moves fastest; the author: one episode.
2. An episodic omega with the gate kinematic whose free circle reaches round the origin somewhere in the record
   (rho_max > 1) is non_resonant, kinematic, when inside every librating episode the median local rho < 1 and the
   free vector turns about the forced one once per cycle (624 Hektor).

Expectations are computed here by hand from the filter gain and the geometry of two vectors, not by the package.
"""

import numpy as np
import pytest

from resonances.data.const import PLANETARY_FREQUENCIES
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.secular import GateResult, LongPeriodSeries, decide, preprocess
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, Reason, ResonanceType, Status
from resonances.resonance.analysis.zlk import analyse_zlk
from resonances.secular import free_elements as fe

SETTINGS = settings_from_config()
AS = 1296000.0
RAD = 2 * np.pi / AS  # rad/yr per arcsec/yr
G5 = PLANETARY_FREQUENCIES['g5']


def _gain(period: float, cutoff: float) -> float:
    """Forward-backward Butterworth order 2: G = 1 / (1 + (P_c / P)^4) (the .env.dist FILTER_ORDER comment)."""
    return 1.0 / (1.0 + (cutoff / period) ** 4)


def _zlk_omega(t: np.ndarray, omega: np.ndarray) -> LongPeriodSeries:
    one = np.ones_like(t)
    return LongPeriodSeries(ResonanceType.LIDOV_KOZAI, 'LK', 'synthetic', t, omega, 5.2 * one, 0.05 * one, 0.3 * one)


def _prominence(series: LongPeriodSeries):
    cutoff = preprocess.choose_cutoff(series, SETTINGS)
    angle = preprocess.filtered_angle(series, cutoff, SETTINGS)
    assert angle is not None
    return cutoff, angle


# ------------------------------------------------------------------------------------------------ 1. prominence


def test_a_libration_near_the_floor_does_not_raise_its_own_threshold():
    # 591986's numbers: P = 6100 yr, half-amplitude 16 deg, ZLK P_c = 2 x 937 = 1874 yr, P_min = 2 P_c = 3748 yr
    t = np.arange(0.0, 500_001.0, 100.0)
    series = _zlk_omega(t, np.radians(288.0 + 16.0 * np.sin(2 * np.pi * t / 6100.0)))
    cutoff, angle = _prominence(series)
    assert cutoff.cutoff_yr == pytest.approx(1874.0, rel=1e-3)
    assert angle.period_floor_yr == pytest.approx(3748.0, rel=1e-3)
    # what the old rule read: the image of the libration, 0.116 x 16 = 1.85 deg, x 4 = 7.4 deg > the 5-deg floor
    image = (_gain(6100.0, 1874.0) - _gain(6100.0, 3748.0)) * 16.0
    assert image == pytest.approx(1.85, abs=0.02)
    assert SETTINGS.zlk_tp_ripple_factor * image > SETTINGS.zlk_tp_prominence_deg
    # now: nothing faster than P_min is left, the floor binds
    assert np.degrees(cutoff.ripple_rad) < 0.2
    assert cutoff.prominence_binding == 'floor'
    assert np.degrees(cutoff.prominence_rad) == pytest.approx(SETTINGS.zlk_tp_prominence_deg)


def test_a_wave_faster_than_the_floor_still_sets_the_threshold():
    # a 3-kyr wave of 3 deg alone: faster than P_min (3748 yr), no libration run, no image; it keeps
    # (G(3000; 1874) - G(3000; 3748)) x 3 = 0.577 x 3 = 1.73 deg in sigma_f - lowpass(sigma, P_min)
    t = np.arange(0.0, 500_001.0, 100.0)
    series = _zlk_omega(t, np.radians(100.0 + 3.0 * np.sin(2 * np.pi * t / 3000.0)))
    cutoff, _ = _prominence(series)
    expected = (_gain(3000.0, 1874.0) - _gain(3000.0, 3748.0)) * 3.0
    assert expected == pytest.approx(1.73, abs=0.02)
    assert np.degrees(cutoff.ripple_rad) == pytest.approx(expected, rel=0.05)
    assert cutoff.prominence_binding == 'ripple'


def test_a_libration_on_a_fast_moving_centre_stays_one_episode():
    # 16 deg at 6.1 kyr on a centre that swings +-40 deg with a 54-kyr sawtooth-like wave (a sine here): the centre
    # rate up to 2 pi 40 / 54000 = 4.7e-3 deg/yr takes 4.7 x 3.05 = 14 deg off a half-swing of 32 deg on the descending
    # side, leaving 18 deg (turns below 17.4 deg were lost under the old threshold); now one episode covers the record
    t = np.arange(0.0, 1_000_001.0, 100.0)
    omega = 288.0 + 40.0 * np.sin(2 * np.pi * t / 54_000.0) + 16.0 * np.sin(2 * np.pi * t / 6100.0)
    analysis = analyse_zlk(_zlk_omega(t, np.radians(omega)), SETTINGS)
    accepted = [episode for episode in analysis.geometry.episodes if episode.accepted]
    assert len(accepted) == 1
    assert analysis.verdict.angle_status == AngleStatus.LIBRATION


# ------------------------------------------------------------------------------------------ 2. episode pinning


def _decide(angle_status, rho_max, rho_episodes, turns, outcome='kinematic'):
    gate = GateResult(
        outcome,
        n_cycles=127.0,
        min_cycles=3.0,
        rho_max=rho_max,
        rho_episodes=rho_episodes,
        free_turns_episodes=turns,
        free_turns_min=SETTINGS.zlk_episode_free_turns_per_cycle,
    )
    return decide.decide_status(ResonanceType.LIDOV_KOZAI, decide.AngleDecision(angle_status), gate, Ledger.from_settings(SETTINGS))


@pytest.mark.parametrize(
    'rho_episodes, turns, status',
    [
        (0.99, 0.94, Status.NON_RESONANT),  # 624 Hektor: the episodes are the forced vector's
        (1.20, 0.94, Status.TRANSIENT),  # the free circle encloses the origin inside an episode: not explained
        (0.99, 0.10, Status.TRANSIENT),  # the free omega librates inside an episode (2023 RS2: 0.15)
        (float('nan'), float('nan'), Status.TRANSIENT),  # not measured: the record-wide rule only
    ],
)
def test_an_episodic_omega_is_kinematic_only_when_every_episode_is_pinned(rho_episodes, turns, status):
    decision = _decide(AngleStatus.EPISODIC, 2.02, rho_episodes, turns)
    assert decision.status == status
    if status == Status.NON_RESONANT:
        assert decision.reason == Reason.KINEMATIC_LIBRATION


def test_the_episode_rule_is_for_episodes_only():
    # a drift has no episodes: rho_max > 1 keeps it uncertain whatever the (stray) episode numbers say
    assert _decide(AngleStatus.DRIFT, 2.02, 0.5, 1.0).status == Status.UNCERTAIN
    # a confirmed gate is not kinematic: the episode numbers change nothing
    assert _decide(AngleStatus.EPISODIC, 2.02, 0.5, 1.0, outcome='confirmed').status == Status.TRANSIENT


def _two_vectors():
    """Forced 0.046 at g5 and free 0.055 at 340"/yr, the free radius 0.030 for 150-200 kyr (ratio 1.20 outside, 0.65
    inside), the free node at -30"/yr: the free omega runs at 370"/yr, the osculating cycle (the free-forced beat) is
    1296000 / (340 - g5) = 3860 yr."""
    t = np.arange(0.0, 500_001.0, 100.0)
    radius = np.where((t >= 150_000.0) & (t <= 200_000.0), 0.030, 0.055)
    z_e = 0.046 * np.exp(1j * G5 * RAD * t) + radius * np.exp(1j * 340.0 * RAD * t)
    Omega = -30.0 * RAD * t + 0.4
    elements = fe.free_elements(t, np.abs(z_e), np.full_like(t, 0.3), Omega, np.angle(z_e) - Omega)
    assert elements.b_omega == pytest.approx(370.0, rel=0.01)
    return elements


BEAT = AS / (340.0 - G5)


def test_episode_pinning_reads_the_ratio_and_the_turns_per_whole_cycle():
    elements = _two_vectors()
    start, cycles = 160_000.0, 5
    rho, per_cycle = elements.episode_pinning(start, start + 5.5 * BEAT, start + cycles * BEAT, cycles)
    assert rho == pytest.approx(0.030 / 0.046, rel=0.05)
    # 370"/yr over five cycles of 3860 yr: 370 x 3860 / 1296000 = 1.10 turns per cycle
    assert per_cycle == pytest.approx(370.0 * BEAT / AS, rel=0.05)
    times, ratio = elements.local_rho(BEAT, 1.0)
    assert len(times) == len(ratio) and np.median(ratio[(times > 300_000.0) & (times < 450_000.0)]) == pytest.approx(0.055 / 0.046, rel=0.1)


def test_episode_pinning_needs_the_free_elements_inside_the_episode():
    from dataclasses import replace

    elements = _two_vectors()
    start = 160_000.0
    args = (start, start + 3 * BEAT, start + 2 * BEAT, 2)
    assert np.isfinite(elements.episode_pinning(*args)[0])
    # one masked sample inside: no measurement there (Codex W26 #1: a phase on 10 % of an episode authorised it)
    keep = elements.keep_e.copy()
    keep[np.searchsorted(elements.times, start + BEAT)] = False
    assert np.isnan(replace(elements, keep_e=keep).episode_pinning(*args)).all()
    # an episode running past the free elements' baseline
    end = elements.times[-1]
    assert np.isnan(elements.episode_pinning(end - BEAT, end + BEAT, end + 0.5 * BEAT, 1)).all()
    # sampled below two samples per cycle
    assert np.isnan(elements.episode_pinning(start, start + 900.0, start + 800.0, 1)).all()


def test_episode_pinning_does_not_borrow_the_ratio_of_the_surroundings():
    # Codex W26 #2: a short episode with the free circle around the origin (ratio 2) inside a stretch at 0.5: the
    # instantaneous median is 2, whatever a window as long as a forced beat would read
    from dataclasses import replace

    elements = _two_vectors()
    t = elements.times
    forced = 0.046 * np.exp(1j * G5 * RAD * t)
    radius = np.where((t >= 200_000.0) & (t <= 210_000.0), 0.092, 0.023)
    patched = replace(elements, z_e_forced=forced, z_e_free=radius * np.exp(1j * 340.0 * RAD * t))
    rho, _ = patched.episode_pinning(200_000.0, 210_000.0, 200_000.0 + 2 * BEAT, 2)
    assert rho == pytest.approx(2.0, rel=0.01)


def test_a_librating_free_omega_makes_no_turns_over_whole_cycles():
    # Codex W26 #5: a bounded free omega of +-170 deg passes 0.94 turns between two points of a part cycle; over whole
    # cycles of its own libration (the osculating omega librates with it) it returns: 0 turns
    from dataclasses import replace

    elements = _two_vectors()
    t = elements.times
    period = 20_000.0
    node = -30.0 * RAD * t
    omega_free = np.radians(170.0) * np.cos(2 * np.pi * t / period)
    patched = replace(elements, z_e_free=0.03 * np.exp(1j * (omega_free + node)), z_i_free=0.2 * np.exp(1j * node))
    _, per_cycle = patched.episode_pinning(100_000.0, 100_000.0 + 3 * period, 100_000.0 + 3 * period, 3)
    assert per_cycle == pytest.approx(0.0, abs=0.01)
