"""W28 B6b (card g08-4, the author's answer A of 2026-10-31): a ripple without measurable physics on an unfinished
slow swing does not decide the angle (rule `ripple_on_arc`, MMR_RIPPLE_ON_ARC).

2004 YG1 9M-10J-4 over 20 kyr: one accepted episode of 5 TPs at 13.47-14.89 kyr, half-amplitude 3.1 deg, its identity
measured and failed (slope/c 0.65, corr 0.79), sits at the one interior turn (13.74 kyr) of a slow centre that sweeps
281 deg (4.91 rad). Below 45 deg neither the force law L4 nor R0 with a can be told from the identity, and the identity
fails: nothing physical says the ripple is a libration of this angle. It becomes an unconfirmed ripple (route
'unconfirmed_ripple', like an F6 forced wave no episode of the angle), and G5a reads the slow centre:
uncertain / insufficient_time (arc), reason baseline_needed, action extend (the card: about 50 kyr, >= 40 kyr).

The synthetic record imitates it independently: c(t) = 270 deg sin(pi t / 28 kyr) (one turn at 14 kyr, 270 deg
= 4.71 rad < 2 pi) plus 3 deg sin(2 pi t / 680 yr) switched on around the turn; the geometry is measured for real
(`geometry.measure_geometry`), only the episode's response (identity) is stated. The numbers that pin it were read off
the measured geometry on 2026-10-31: the accepted run 13.2-14.4 kyr, 5 TPs, 3.1 deg, the centre's turn at 14.03 kyr.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import decide
from resonances.resonance.analysis.mmr.models import EpisodeResponse, ForcedWave
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, CheckValue, Reason, Status
from tests.resonances.analysis.builders import make, make_centre, make_episode, make_geometry, make_turning_points
from tests.resonances.analysis.test_w20_regime_cuts import MMR_RULES, ONE_DEG, S, SAMPLING, filtered

CUTOFF_YR = 75.0  # the 2004 YG1 record's cutoff


def swing_record(ripple_deg=3.0, swing_deg=270.0):
    """A slow swing of `swing_deg` that turns back at 14 kyr, with a ripple of `ripple_deg` of 680 yr around the turn."""
    t = np.arange(0.0, 20e3, 5.0)
    centre = math.radians(swing_deg) * np.sin(np.pi * t / 28e3)
    switched_on = np.exp(-0.5 * ((t - 14e3) / 700.0) ** 8)
    sigma = centre + math.radians(ripple_deg) * switched_on * np.sin(2 * np.pi * t / 680.0)
    angle = filtered(t, sigma, CUTOFF_YR, **MMR_RULES)
    return angle, g.measure_geometry(angle, ONE_DEG, S)


def response_of(run, identity, slope=0.65, corr=0.79):
    """The episode's measured response: only the identity (L3) matters here (YG1's slope/c and correlation)."""
    return make(
        EpisodeResponse,
        episode_id=run.episode_id,
        n_tp=run.n_tp,
        period_yr=run.period_med_yr,
        identity=identity,
        identity_slope_ratio=slope,
        identity_corr=corr,
    )


def set_aside(angle, geometry, responses, settings=S):
    """What the analysis does before G2 (`analyse._without_forced_waves`): the waves go to `Geometry.forced_waves`."""
    waves = decide.forced_wave_episodes(responses, settings, geometry)
    if waves:
        geometry.forced_waves = waves
        g.refresh_edges(geometry, angle, settings)
    return waves


def decide_angle(geometry):
    ledger = Ledger.from_settings(S)
    return decide.decide_angle(SAMPLING, False, geometry, None, S, ledger), ledger


def test_the_synthetic_record_is_a_ripple_at_the_turn_of_a_slow_swing():
    _, geometry = swing_record()
    (run,) = [r for r in geometry.runs if r.accepted]
    assert (run.n_tp, round(math.degrees(run.half_amp_max_rad), 1)) == (5, 3.1)
    assert run.t_start_yr == pytest.approx(13215.0, abs=50) and run.t_end_yr == pytest.approx(14440.0, abs=50)
    assert decide.interior_turn_times(geometry) == pytest.approx([14030.0], abs=50)
    assert geometry.centre.range_rad == pytest.approx(math.radians(270.0), abs=0.15)
    assert geometry.drift_fraction == 0.0  # no drift run: without the rule G6 reads the ripple's TPs


def test_a_failing_three_degree_ripple_on_a_slow_swing_gives_insufficient_time_and_extend():
    angle, geometry = swing_record()
    (run,) = [r for r in geometry.runs if r.accepted]
    before, _ = decide_angle(geometry)
    assert (before.angle_status, before.rule) == (AngleStatus.EPISODIC, 'G2')  # the ripple decided the angle
    (wave,) = set_aside(angle, geometry, [response_of(run, CheckValue.FAIL)])
    assert (wave.episode_id, wave.route, wave.ripple) == (run.episode_id, 'unconfirmed_ripple', True)
    assert geometry.episodes == []
    decision, ledger = decide_angle(geometry)
    assert (decision.angle_status, decision.subtype, decision.rule) == (AngleStatus.INSUFFICIENT_TIME, AngleSubtype.ARC, 'G5a')
    assert any(name.startswith('G2 unconfirmed ripple') for name in (entry.name for entry in ledger))
    status = decide.decide_status(decision, SAMPLING, geometry, None, None, None, None, None, None, S, Ledger.from_settings(S))
    assert (status.status, status.reason) == (Status.UNCERTAIN, Reason.BASELINE_NEEDED)
    # the extension, by hand: the slow arc's bound is twice the longest stretch without a turn of the centre,
    # 2 x (19920 - 14030) < 2 x (14030 - 75) = 27.9 kyr; required = 3/2 P + 2 P_c = 41.9 + 0.15 = 42.0 kyr > T =
    # 19920 - 75 = 19.8 kyr, so N = round_up(max(42.0, 2 T = 39.7)) = 50 kyr; the next check N8,
    # round_up(max(2 T, min(N, 4 T))), is 50 kyr too (the card: about 50 kyr, >= 40 kyr; YG1 itself: 50 kyr)
    span = geometry.window[1] - geometry.window[0]
    base = decide.baseline(geometry, CUTOFF_YR, span, float('nan'), S, decision)
    assert (base.period_source, base.period_yr) == ('arc_slow', pytest.approx(27.9e3, abs=150))
    assert base.required_yr == pytest.approx(1.5 * base.period_yr + 2 * CUTOFF_YR) and base.check == CheckValue.FAIL
    assert decide.extend_years(status.status, decision.angle_status, base, span, S) == 50e3


def test_a_small_libration_whose_identity_holds_stays():
    angle, geometry = swing_record()
    (run,) = [r for r in geometry.runs if r.accepted]
    assert set_aside(angle, geometry, [response_of(run, CheckValue.PASS, slope=1.02, corr=0.97)]) == ()
    decision, _ = decide_angle(geometry)
    assert (decision.angle_status, decision.rule, decision.episode_id) == (AngleStatus.EPISODIC, 'G2', run.episode_id)


def test_an_unmeasured_identity_confirms_nothing_either():
    """W32 (the author's 2002 BG card C): a ripple whose identity cannot be measured (one cycle) is as unconfirmed as
    one whose identity fails; only a passing identity keeps a small run an episode of the angle."""
    angle, geometry = swing_record()
    (run,) = [r for r in geometry.runs if r.accepted]
    (wave,) = set_aside(angle, geometry, [response_of(run, CheckValue.NOT_MEASURED)])
    assert wave.ripple and wave.episode_id == run.episode_id


def test_at_45_degrees_and_above_a_swing_is_never_a_ripple():
    """A 50-deg loop at the same turn: L4 and R0 can be measured, so a failed identity alone sets nothing aside."""
    angle, geometry = swing_record(ripple_deg=50.0)
    accepted = [r for r in geometry.runs if r.accepted]
    assert accepted and all(math.degrees(r.half_amp_max_rad) >= S.mmr_a_response_min_amp_deg for r in accepted)
    assert set_aside(angle, geometry, [response_of(r, CheckValue.FAIL) for r in accepted]) == ()


def test_the_setting_off_sets_nothing_aside():
    angle, geometry = swing_record()
    (run,) = [r for r in geometry.runs if r.accepted]
    off = replace(S, mmr_ripple_on_arc=False)
    assert set_aside(angle, geometry, [response_of(run, CheckValue.FAIL)], off) == ()


# --- the slow centre and the episode decide whether a swing is unfinished (built geometries) -------------------------


def ripple_geometry(turns=(14030.0,), range_rad=4.7, coorbital=False, covering=False, extra_tps=()):
    """A window 0-20 kyr with one accepted 3-deg ripple of 5 TPs at 13.2-14.4 kyr; the slow centre (W_c 2 kyr) turns
    at `turns` and spans `range_rad`; `extra_tps`: turning points outside the ripple."""
    tps = np.array([13215.0, 13520.0, 13830.0, 14135.0, 14440.0])
    gaps = dict(coverage_start_gap_p=0.0, coverage_end_gap_p=0.0) if covering else {}
    run = make_episode(episode_id=0, tp_times_yr=tps, half_amp_max_rad=math.radians(3.0), period_med_yr=610.0, **gaps)
    centre = make_centre(turns=make_turning_points(turns), range_rad=range_rad, window_yr=2000.0, max_pass_rad=range_rad)
    times = np.sort(np.concatenate([tps, np.asarray(extra_tps, float)]))
    return make_geometry(window=(0.0, 20e3), runs=[run], centre=centre, coorbital=coorbital, turning_points=make_turning_points(times)), run


@pytest.mark.parametrize(
    'kwargs, why',
    [
        (dict(range_rad=7.0), 'the centre circulates: a revolution, no unfinished swing'),
        (dict(turns=()), 'the centre never turns back inside the record'),
        (dict(turns=(5e3, 9e3, 13e3)), 'three turns: a whole slow cycle is seen (2003 UC22 16M-27)'),
        (dict(turns=(500.0,)), 'a turn within W_c of the edge is the low-pass bending'),
        (dict(covering=True), 'an episode covering the record is the angle itself, no ripple on an arc'),
        (dict(coorbital=True), '1:1: its law is measured at any amplitude'),
    ],
)
def test_no_unfinished_slow_swing_no_ripple(kwargs, why):
    geometry, run = ripple_geometry(**kwargs)
    assert decide.ripple_waves(geometry, [response_of(run, CheckValue.FAIL)], S) == (), why


def test_two_turns_are_still_an_unfinished_swing():
    """Two interior turns show one half-swing between them, not a whole slow cycle (523683 2U-9 over 100 kyr)."""
    geometry, run = ripple_geometry(turns=(6e3, 14030.0), range_rad=4.1)
    (wave,) = decide.ripple_waves(geometry, [response_of(run, CheckValue.FAIL)], S)
    geometry.forced_waves = (wave,)
    decision = decide._g4_to_g6(SAMPLING, geometry, S, Ledger.from_settings(S))
    # two turns are no G5a arc; the slow centre's 4.1 rad >= pi: an arc by G6
    assert (decision.angle_status, decision.subtype, decision.rule) == (AngleStatus.INSUFFICIENT_TIME, AngleSubtype.ARC, 'G6')


def test_turning_points_outside_the_ripples_are_swings_of_the_angle():
    geometry, run = ripple_geometry(extra_tps=(3000.0, 3400.0))
    geometry.forced_waves = decide.ripple_waves(geometry, [response_of(run, CheckValue.FAIL)], S)
    assert decide.tp_outside_ripples(geometry) == 2
    decision = decide._g4_to_g6(SAMPLING, geometry, S, Ledger.from_settings(S))
    assert (decision.angle_status, decision.subtype, decision.rule) == (AngleStatus.INSUFFICIENT_TIME, AngleSubtype.FEW_CYCLES, 'G6')


def test_a_ripple_is_neither_an_f6_wave_nor_a_strength_mark():
    geometry, run = ripple_geometry()
    geometry.forced_waves = decide.ripple_waves(geometry, [response_of(run, CheckValue.FAIL)], S)
    assert decide.forced_wave_columns(geometry)['geo_n_forced_wave_episodes'] == 0
    assert geometry.excluded_ids == frozenset({0})
    both = ForcedWave(0, 5, 610.0, 0.65, 0.79, '', float('nan'), False, True, ripple=True)
    assert both.route == 'lattice_line;unconfirmed_ripple'
