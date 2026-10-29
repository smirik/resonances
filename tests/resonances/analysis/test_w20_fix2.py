"""W20 FIX2: the phase return at any level of the swing (LIBRATION_RETURN_LEVELS) and the onset test O1 only where a
capture is claimed.

Expected numbers come from the construction of each record:
- the cosine cycle: sigma = A cos(2 pi (t - t1) / P), A = 150 deg, P = 400 yr, t1 = 56 yr, on [0, 436] yr (from 0.14 P
  before its maximum to 0.95 P after it). The record starts at sigma = A cos(2 pi x 0.14) = 0.637 A, above the
  mid-level 0 of the two turning points (+A at t1, -A at t1 + P/2): sigma never crosses the mid-level before t1. Of the
  levels v2 + k (v1 - v2) / 20 only k = 17, 18, 19 (0.85, 0.90, 0.95 of the swing, above 0.819) are crossed upwards
  before t1, all in the same state; k = 18 (sigma = 0.8 A) at t1 - acos(0.8) P / (2 pi) = t1 - 0.10242 P = 15.03 yr,
  and again after the minimum at t1 + 0.89758 P = 415.03 yr: t_b - t_a = P.
- the drifting centre: sigma = 2 pi t / T + B sin(2 pi t / 4.4 kyr) (B = 15, 30, 60 deg; T = 20, 40 kyr, and 10 kyr for
  B >= 30 deg); the loops cross every level forth at the speed of the drift plus the loop's and back at the loop's
  minus the drift's, so condition 6 (the same speed both ways) fails at every level.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.filtering import edge_window, lowpass
from resonances.resonance.analysis.mmr import decide, io
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.decide import StatusDecision
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import LibrationOrigin, OriginDetail, Reason, Status

S = settings_from_config()
MID_ONLY = replace(S, libration_return_levels=0)
ONE_DEG = math.radians(1.0)
NAN = float('nan')
C = -30.0


def filtered(t, angle, cutoff, slow):
    angle_f = lowpass(t, angle, cutoff, 2)
    return g.FilteredAngle(
        t_yr=t,
        angle_raw=angle,
        angle_f=angle_f,
        rate_f=np.gradient(angle_f, t),
        angle_f_half=lowpass(t, angle, cutoff / 2, 2),
        angle_f_double=lowpass(t, angle, 2 * cutoff, 2),
        slow_f=lowpass(t, slow, cutoff, 2),
        identity_coefficient=C,
        cutoff_yr=cutoff,
        step_yr=float(t[1] - t[0]),
        window=edge_window(t, cutoff),
        return_closure=True,
    )


def cosine_cycle(period=400.0, amplitude_deg=150.0, t1=56.0):
    t = np.arange(0.0, t1 + 0.95 * period, 0.25)
    omega, amplitude = 2 * np.pi / period, math.radians(amplitude_deg)
    sigma = amplitude * np.cos(omega * (t - t1))
    return t, sigma, -amplitude * omega * np.sin(omega * (t - t1)) / C


def test_a_cycle_that_never_crosses_its_mid_level_returns_at_another_level():
    t, sigma, slow = cosine_cycle()
    angle = filtered(t, sigma, 5.0, slow)
    tp = g.measure_geometry(angle, ONE_DEG, S).turning_points
    assert len(tp) == 2
    assert np.asarray(tp.times_yr) == pytest.approx([56.0, 256.0], abs=1.0)
    pair = np.array([0, 1])
    assert g.return_cycle(angle, tp, pair) is None  # no crossing of the mid-level before the maximum
    assert g.phase_return(angle, tp, pair, MID_ONLY) is None
    cycle = g.phase_return(angle, tp, pair, S)
    assert cycle is not None and cycle.level_fraction == pytest.approx(0.9)
    assert math.degrees(cycle.level_rad) == pytest.approx(0.8 * 150.0, abs=1.0)
    assert (cycle.t_a_yr, cycle.t_b_yr) == (pytest.approx(15.03, abs=1.0), pytest.approx(415.03, abs=1.0))
    assert cycle.period_yr == pytest.approx(400.0, abs=1.0)
    assert g.return_mismatch(cycle, S) <= 1.0


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'example, expected',
    [
        ('coorbital__2020_CB__1E-1+0+0', ('transient', 'episodic', 'low')),
        ('mmr3__60608__1U-1N-2+0+0+2', ('non_resonant', 'circulation', 'medium')),
    ],
)
def test_a_return_at_one_isolated_level_is_no_cycle(example, expected):
    """2020 CB (turning points at 110 and 190 yr, a swing of 312 deg) closes only at 0.90 of it, next to the first
    turn; 60608 1U-1N-2 (a 67-deg back-step of a slow circulation at 5.5-11.2 kyr) only at 0.70. Neither has a closing
    neighbour on the grid of 1/20 (FIX2 level probe), so neither run is a cycle and both verdicts stay as at FIX1."""
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    result = analyse_mmr(io.load_series(example), S)
    v = result.verdict
    assert (str(v.status), str(v.angle_status), str(v.confidence)) == expected
    assert not any(run.closed_by_return for run in result.geometry.runs)


def test_the_closed_cycle_records_its_level():
    t, sigma, slow = cosine_cycle()
    geometry = g.measure_geometry(filtered(t, sigma, 5.0, slow), ONE_DEG, S)
    (run,) = [run for run in geometry.runs if run.n_tp == 2]
    assert run.accepted and run.closed_by_return
    assert run.return_level_fraction == pytest.approx(0.9)
    assert run.period_med_yr == pytest.approx(400.0, abs=1.0)
    mid_only = g.measure_geometry(filtered(t, sigma, 5.0, slow), ONE_DEG, MID_ONLY)
    (run,) = [run for run in mid_only.runs if run.n_tp == 2]
    assert not run.accepted and not run.closed_by_return


@pytest.mark.parametrize(
    'loop_deg, drift_kyr', [(15.0, 20.0), (15.0, 40.0), (30.0, 10.0), (30.0, 20.0), (30.0, 40.0), (60.0, 10.0), (60.0, 20.0)]
)
def test_loops_of_a_drifting_centre_close_at_no_level(loop_deg, drift_kyr):
    t = np.arange(0.0, 100e3, 10.0)
    sigma = 2 * np.pi * t / (drift_kyr * 1e3) + math.radians(loop_deg) * np.sin(2 * np.pi * t / 4400.0)
    angle_f = lowpass(t, sigma, 500.0, 2)
    angle = filtered(t, sigma, 500.0, np.gradient(angle_f, t) / C)
    tp = g.measure_geometry(angle, ONE_DEG, S).turning_points
    assert len(tp) > 40
    assert not any(g.phase_return(angle, tp, np.array([i, i + 1]), S) for i in range(len(tp) - 1))


def codex_drift(cutoff, drift=0.2):
    """Codex's case (review B #2): sigma = cos(2 pi t / 400) + drift t / 400 [rad] on [-140, 308) yr, a drifting centre
    of 0.2 rad (11.5 deg) per 400-yr cycle, a in step with sigma's rate."""
    t = np.arange(-140.0, 308.0, 0.25)
    sigma = np.cos(2 * np.pi * t / 400) + drift * t / 400
    return t, filtered(t, sigma, cutoff, np.gradient(sigma, t) / C)


@pytest.mark.parametrize('cutoff', [2.0, 10.0])
def test_codex_drifting_centre_closes_only_in_a_band_of_two(cutoff):
    """At cutoffs 2 and 10 yr the two turning points of Codex's case close at 0.55 and 0.60 of the swing only (not at
    the mid-level): a band of two levels. The default band (2: what 164207 and 2025 RB7 at 1 kyr need, 0.15-0.20 and
    0.20-0.25) keeps it at 0.60, the smaller mismatch; a band of 3 rejects it. A single cycle with a slowly drifting
    centre may be a libration about a moving centre: the rule does not decide that, it only asks for a band."""
    t, angle = codex_drift(cutoff)
    tp = g.measure_geometry(angle, ONE_DEG, S).turning_points
    assert len(tp) == 2
    pair = np.array([0, 1])
    i1, i2 = (int(i) for i in np.asarray(tp.indices))
    samples = g.return_samples(t, (i1, i2), None, S.libration_return_period_tol)
    closing = [k for k in range(1, 20) if (c := g.return_cycle(angle, tp, pair, None, k / 20, samples)) is not None and g.closes(c, S)]
    assert closing == [11, 12]
    assert g.phase_return(angle, tp, pair, S).level_fraction == pytest.approx(0.6)
    assert g.phase_return(angle, tp, pair, replace(S, libration_return_band_levels=3)) is None


def test_the_mid_level_needs_a_closing_neighbour_except_alone():
    """closing_band counts the consecutive closing levels. Levels 0.25, 0.5, 0.75: the mid-level closing alone is no
    cycle on a grid (the neighbour condition applies to it too) and is one with no grid (LIBRATION_RETURN_LEVELS = 0,
    the CS4 rule); with a closing neighbour it is kept whatever the neighbour's mismatch. Off the mid-level, a band of
    two qualifies under the default and not under a band of 3; the smaller mismatch wins inside the bands."""
    assert g.closing_band([False, True, True, False, True, False, True, True, True]) == [0, 2, 2, 0, 1, 0, 3, 3, 3]
    fractions = [0.25, 0.5, 0.75]
    assert g.return_level(fractions, [False, True, False], [NAN, 0.9, NAN], False, 2) is None
    assert g.return_level(fractions, [False, True, False], [NAN, 0.9, NAN], True, 2) == 1
    assert g.return_level(fractions, [False, True, True], [NAN, 0.9, 0.1], False, 2) == 1
    fractions = [0.2, 0.25, 0.3, 0.5, 0.7]
    assert g.return_level(fractions, [True, True, False, False, True], [0.6, 0.4, NAN, NAN, 0.1], False, 2) == 1
    assert g.return_level(fractions, [True, True, False, False, True], [0.6, 0.4, NAN, NAN, 0.1], False, 3) is None


def test_the_return_search_stays_within_the_cycle_window():
    """Turning points at 100 and 300 yr (Delta = 200 yr), period tolerance 0.25: a crossing farther than
    1.25 x 400 = 500 yr from them (before -400 or after 800 yr) would make (t_b - t_a) / (2 Delta) > 1.75: the samples
    searched are [-400, 800] yr inside a record [-1000, 1000] yr, and the record's own bounds when narrower."""
    t = np.arange(-1000.0, 1000.5, 0.5)
    i1, i2 = int(np.searchsorted(t, 100.0)), int(np.searchsorted(t, 300.0))
    lo, hi = g.return_samples(t, (i1, i2), None, 0.25)
    assert (t[lo], t[hi]) == (-400.0, 800.0)
    lo, hi = g.return_samples(t, (i1, i2), (0.0, 500.0), 0.25)
    assert (t[lo], t[hi]) == (0.0, 500.0)


@pytest.mark.parametrize('name', ['libration_return_tol', 'libration_return_period_tol'])
def test_the_return_tolerances_must_be_positive(name):
    from resonances.resonance.analysis.settings import validate_settings

    with pytest.raises(ValueError, match='positive'):
        validate_settings(replace(S, **{name: 0.0}))


# --- O1 only where a capture is claimed -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    'origin, detail, claimed',
    [
        (LibrationOrigin.DYNAMIC, OriginDetail.NONE, True),
        (LibrationOrigin.UNKNOWN, OriginDetail.NONE, False),  # W21: non_resonant of unknown origin claims no capture
        (LibrationOrigin.SUBRESONANCE, OriginDetail.FIXED_SUBRESONANCE, True),
        (LibrationOrigin.INDUCED, OriginDetail.OWNER_LIBRATION, True),
        (LibrationOrigin.INDUCED, OriginDetail.OWNER_FORCED_WAVE, False),
        (LibrationOrigin.KINEMATIC, OriginDetail.VARPI_TERM, False),
        (LibrationOrigin.KINEMATIC, OriginDetail.SECULAR_CYCLE, False),
    ],
)
def test_onset_is_a_test_of_a_capture(origin, detail, claimed):
    decision = StatusDecision(Status.NON_RESONANT, Reason.INDUCED_LIBRATION, origin, str(detail), '')
    assert decide.capture_claimed(decision) is claimed


@pytest.mark.benchmark
def test_385266_forced_wave_is_no_episode_and_the_owner_names_it():
    """385266 16N-27 (author 2026-10-19: non_resonant/episodic/medium, "55-100 kyr cycles = forced wave"). Before W20
    CS7 the episodes at 56-68 and 68-99 kyr decided (episodic/low: L3 0.823, L5 4.48 of the decisive one). Since CS7
    they are forced waves (F6): their periods sit on the lattice lines 3N-5 (4444 yr) and 6N-10 (1695 yr) and their
    identity fails (slope 0.72 c, corr 0.82); without them sigma circulates and the owner 3N-5+0+2 still names the
    wave: non_resonant/circulation/medium. W20 round C: the word is the geometry's with the forced waves (their loops
    are librations of sigma), episodic as the author's; status, reason and confidence stay those without them."""
    example = 'mmr2__385266__16N-27+0+11'
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    analysis = analyse_mmr(io.load_series(example), S)
    v = analysis.verdict
    assert (str(v.status), str(v.angle_status), str(v.origin_detail), v.owner) == (
        'non_resonant',
        'episodic',
        'owner_forced_wave',
        '3N-5+0+2',
    )
    assert sorted(w.force_line for w in analysis.geometry.forced_waves) == ['3N-5', '6N-10']
    assert 'onset_failed' not in v.caps and str(v.confidence) == 'medium' and str(v.reason) == 'circulation'


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'example, expected, fraction',
    [
        ('coorbital__2025_RB7__1E-1+0+0', ('resonant', 'libration', 'medium'), 0.2),
        ('coorbital__164207__1E-1+0+0', ('resonant', 'libration', 'high'), 0.2),
        ('coorbital__2016_CA138__1E-1+0+0', ('transient', 'episodic', 'medium'), 0.5),
        ('coorbital__3753__1E-1+0+0', ('resonant', 'libration', 'medium'), 0.5),
    ],
)
def test_benchmark_returns_at_their_levels(example, expected, fraction):
    """2025 RB7 at 1 kyr (author resonant/medium; the E report's return 72 -> 782 yr on the raw angle): at the mid-level a
    is 6.8 % and the rate 22 % off, at 0.20 and 0.25 of the swing (62 -> 771, 74 -> 784 yr) both within 3 %. 164207
    (author resonant/high): sigma never crosses the mid-level of the horseshoe turns (752, 932 yr) before 752 yr; at
    0.15 and 0.20 of the swing it does (611 -> 973 yr at 0.20, the rate within 0.6 %). CA138 and 3753 keep their
    mid-level returns. The kept level needs a closing neighbour on the grid of 1/20."""
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    result = analyse_mmr(io.load_series(example), S)
    v = result.verdict
    assert (str(v.status), str(v.angle_status), str(v.confidence)) == expected
    returned = [run for run in result.geometry.runs if run.closed_by_return and run.accepted]
    assert [run.return_level_fraction for run in returned] == [pytest.approx(fraction)]
