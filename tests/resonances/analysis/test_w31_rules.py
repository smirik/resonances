"""W31 rules (the author's answers to the W30 page, 2026-11-03), each on a case built by hand:

- the forced-wave test against the episode's background (`mmr.background_wave`): a small episode whose sigma and a
  envelopes stay those of the steady wave around it is that wave; one that adds its own swing is not;
- a 1:1 turn within the thin band above MMR_COORBITAL_TURN_ZONE_MAX is undecided, not a kick (2020 FN3);
- a 1:1 swing of one compound well joins two runs (`geometry._compound_swing`, 439908: centres 42 deg apart);
- the conventional period floor is read with its thin band, a co-orbital's measured floor as it is (WK15, 2022 AP1)."""

import math

import numpy as np

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.mmr import background_wave as bw
from resonances.resonance.analysis.mmr import decide
from resonances.resonance.analysis.settings import AnalysisSettings
from tests.resonances.analysis.builders import make_episode, make_geometry

SETTINGS = AnalysisSettings()
PERIOD = 200.0
STEP = 2.0
T = np.arange(0.0, 20000.0, STEP)


def wave(amplitude_deg: float, a_amplitude: float, bump=None):
    """sigma: a steady wave of `amplitude_deg` at PERIOD; a: the same wave scaled; `bump` (t0, t1, factor) multiplies
    both inside [t0, t1] (a libration adding its own swing)."""
    scale = np.ones_like(T)
    if bump is not None:
        t0, t1, factor = bump
        scale[(T >= t0) & (T <= t1)] = factor
    phase = 2 * math.pi * T / PERIOD
    return np.radians(amplitude_deg) * scale * np.sin(phase), a_amplitude * scale * np.cos(phase)


def episode(t0=9000.0, t1=10000.0, half_amp_deg=2.0):
    times = np.linspace(t0, t1, 11)  # 11 turning points half a period apart over 1000 yr
    return make_episode(11, episode_id=3, tp_times_yr=times, period_med_yr=PERIOD, half_amp_max_rad=math.radians(half_amp_deg))


def test_a_small_episode_of_the_steady_wave_is_forced():
    sigma, a = wave(2.0, 1e-4)
    found = bw.measure_background_wave(T, sigma, a, episode(), [episode()], (0.0, T[-1]), SETTINGS)
    # inside and outside the envelope is the same 2 deg: the ratios are 1 within the filter's edge effects
    assert found.outcome == bw.FORCED and found.reason == ''
    assert abs(found.sigma_hi - 1.0) < 0.1 and abs(found.a_hi - 1.0) < 0.1
    # 12 periods each side, less the run +- P/2: (24 - 1) periods of reference
    assert abs(found.reference_periods - 23.0) < 0.5


def test_an_episode_that_adds_its_own_swing_is_not_forced():
    # a libration triples the swing of sigma and a while it lasts
    sigma, a = wave(2.0, 1e-4, bump=(9000.0, 10000.0, 3.0))
    found = bw.measure_background_wave(T, sigma, a, episode(), [episode()], (0.0, T[-1]), SETTINGS)
    assert found.outcome == bw.OWN and found.sigma_hi > 2.0 and found.a_hi > 2.0


def test_a_noisy_background_is_not_measured():
    rng = np.random.default_rng(1)
    sigma, a = rng.normal(0, 0.03, T.size), rng.normal(0, 1e-4, T.size)
    found = bw.measure_background_wave(T, sigma, a, episode(), [episode()], (0.0, T[-1]), SETTINGS)
    assert (found.outcome, found.reason) == (bw.NOT_MEASURED, 'incoherent')


def test_only_small_episodes_of_a_non_coorbital_angle_are_tested():
    sigma, a = wave(2.0, 1e-4)
    small, large = episode(), episode(t0=3000.0, t1=4000.0, half_amp_deg=60.0)
    assert [w.episode_id for w in bw.background_waves(T, sigma, a, [small, large], (0.0, T[-1]), SETTINGS)] == [3]
    assert bw.background_waves(T, sigma, a, [small], (0.0, T[-1]), SETTINGS, coorbital=True) == ()


def test_a_turn_in_the_thin_band_of_the_zone_is_undecided():
    # zone max 1.1, thin band 0.1: 1.11 and 1.15 lie below 1.21; out/in 1.0 is a mirror
    assert SETTINGS.mmr_coorbital_turn_zone_max == 1.1
    near = g.LegTurn(100.0, 'planet', out_in=1.0, zone=1.15)
    assert not decide.turn_reflects(near, SETTINGS) and decide.turn_reflects_within_band(near, SETTINGS)
    assert not decide.turn_reflects_within_band(g.LegTurn(100.0, 'planet', out_in=1.0, zone=1.25), SETTINGS)
    # a kick by energy (255071: out/in 1.83, |ln| 0.60 > 0.34) is no thin-band turn, whatever its zone
    assert not decide.turn_reflects_within_band(g.LegTurn(100.0, 'planet', out_in=1.83, zone=0.78), SETTINGS)


def test_a_compound_swing_keeps_its_centre():
    values = np.radians([10.0, 90.0, -100.0, 30.0, 110.0])
    # i = 1 (90), j = 2 (-100): left centre (10 + 90)/2 = 50, right centre (-100 + 30)/2 = -35: 85 deg apart < 180
    assert g._compound_swing(values, 1, 2)
    # a passage: the right well's centre is a turn on (-100 + 360, 30 + 360): 325 deg apart
    passage = np.radians([10.0, 90.0, 260.0, 390.0, 470.0])
    assert not g._compound_swing(passage, 1, 2)
    assert not g._compound_swing(values, 0, 1)  # no half-cycle before the first turning point


def test_the_conventional_floor_is_read_with_its_thin_band():
    run = episode()
    # the convention MMR_LIBRATION_PERIOD_MIN_YR 150 yr, less the thin band 0.1: 135 yr (2001 WK15: 142.9 yr counts)
    assert decide._eligible_floor(make_geometry(), SETTINGS, run) == 135.0
    # a co-orbital's floor is its planet's tadpole period, a measurement: read as it is (2022 AP1)
    tadpole = make_geometry(period_floor_yr=180.0, period_floor_setting='mmr_coorbital_period_floor_factor')
    assert decide._eligible_floor(tadpole, SETTINGS, run) == 180.0


def run_of(eid: int, t0: float, t1: float):
    times = np.arange(t0, t1 + 1e-9, PERIOD / 2)  # turning points half a period apart
    return make_episode(len(times), episode_id=eid, tp_times_yr=times, period_med_yr=PERIOD, half_amp_max_rad=math.radians(2.0))


def test_a_run_between_two_forced_runs_is_measured_against_them():
    """W32 (2012 DE61 ep 4): runs measured forced are the background. Run 2 (8800-9600 yr) sits between runs 1
    (5000-8600) and 3 (9800-13400): with every accepted run (itself too) masked +- P/2 its +-12 P reference (6400-12000
    yr) keeps nothing (< 4 periods: not measured); runs 1 and 3 keep 2600-4900 and 13500-15800 yr (11.5 P each) and
    are forced. Measured again with runs 1 and 3 in its reference, run 2 is forced too: 6400-8700 and 9700-12000 yr,
    23 periods."""
    sigma, a = wave(2.0, 1e-4)
    runs = [run_of(1, 5000.0, 8600.0), run_of(2, 8800.0, 9600.0), run_of(3, 9800.0, 13400.0)]
    alone = bw.measure_background_wave(T, sigma, a, runs[1], runs, (0.0, T[-1]), SETTINGS)
    assert (alone.outcome, alone.reason) == (bw.NOT_MEASURED, 'short_reference')
    assert alone.reference_periods == 0.0
    waves = bw.background_waves(T, sigma, a, runs, (0.0, T[-1]), SETTINGS)
    assert [(w.episode_id, w.outcome) for w in waves] == [(1, bw.FORCED), (2, bw.FORCED), (3, bw.FORCED)]
    assert abs(waves[1].reference_periods - 23.0) < 0.5


def test_an_own_run_keeps_the_background_without_forced_runs_unchanged():
    # no run forced: one pass, the same measurement as alone
    sigma, a = wave(2.0, 1e-4, bump=(9000.0, 10000.0, 3.0))
    (found,) = bw.background_waves(T, sigma, a, [episode()], (0.0, T[-1]), SETTINGS)
    alone = bw.measure_background_wave(T, sigma, a, episode(), [episode()], (0.0, T[-1]), SETTINGS)
    assert found == alone and found.outcome == bw.OWN
