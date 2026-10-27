"""A libration run must close a full cycle (`geometry.closure_ratios`, W12).

A full cycle (TP_i, TP_i+1, TP_i+2) is closed when sigma returns: its net advance |v_i+2 - v_i| is below the mean of its
two half-swings. Worked out by hand for a maximum 0, a minimum -a and a maximum -a + b: the cycle centre of the run
classifier is (v_i + 2 v_i+1 + v_i+2) / 4 = (-3a + b) / 4, so "0 above it" is b < 3a, "-a + b above it" is a < 3b and
"-a below it" always holds; the net advance |a - b| is below (a + b) / 2 exactly when a / 3 < b < 3 a. An episode
needs one closed cycle; an open edge cycle (entry or exit of a capture) is allowed.

The real case is 1864 3E-5S-5+0+0+7 (benchmark, author: non_resonant): in its last regime the TPs -1611, -1922,
-1917, -2150 deg (half-swings 311, 5, 233 deg) are a circulation that turned back 5 deg once; the bands of its two
averaged cycles overlap by 5 deg, so the band test alone accepted it as a stationary episode (W11)."""

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.mmr import io
from resonances.resonance.analysis.settings import settings_from_config

S = settings_from_config()
RECORD = (0.0, 1.0e6)

EXAMPLE_1864 = 'mmr3__1864__3E-5S-5+0+0+7'
TIMES_1864 = [15393.76, 16153.95, 16318.99, 17274.22]
VALUES_1864 = [-1610.87, -1922.14, -1916.84, -2149.59]


def turning_points(values_deg, half_period_yr=100.0, times=None) -> g.TurningPoints:
    values = np.radians(np.asarray(values_deg, dtype=float))
    times = np.arange(len(values)) * half_period_yr if times is None else np.asarray(times, dtype=float)
    kinds = np.where(np.arange(len(values)) % 2 == 0, np.sign(values[0] - values[1]), np.sign(values[1] - values[0]))
    return g.TurningPoints(np.arange(len(values)), times, values, kinds.astype(int), np.radians(1.0))


def classify(tp: g.TurningPoints) -> g.Episode:
    return g._classify_run(0, tp, np.arange(len(tp)), RECORD, S)


def own_side_of_centre(v0, v1, v2) -> bool:
    """The other formulation, computed directly: the middle TP on one side of the cycle centre, both ends on the other."""
    centre = (v0 + 2 * v1 + v2) / 4
    side = np.sign(v1 - centre)
    return bool(side != 0 and np.sign(v0 - centre) == -side and np.sign(v2 - centre) == -side)


# ------------------------------------------------------------------ the criterion itself


@pytest.mark.parametrize(
    'values, closed',
    [
        ([0.0, -100.0, -20.0], True),  # out 100, back 80: advance 20 < 90
        ([0.0, -90.0, -59.0], True),  # back 31 > 90 / 3: advance 59 < (90 + 31) / 2 = 60.5
        ([0.0, -90.0, -61.0], False),  # back 29 < 90 / 3: advance 61 > (90 + 29) / 2 = 59.5
        ([0.0, -311.27, -305.97], False),  # 1864: out 311, back 5
        ([0.0, 20.0, -80.0], False),  # a 20-deg turn before a 100-deg swing: the first half-swing is the small one
        ([10.0, 340.0, 25.0], True),  # a horseshoe: 330 out, 315 back
    ],
)
def test_closure_by_hand(values, closed):
    assert (g.closure_ratios(np.radians(values)) < 1).tolist() == [closed]
    assert own_side_of_centre(*values) == closed


def test_closure_is_each_tp_on_its_own_side_of_the_cycle_centre():
    rng = np.random.default_rng(20261016)
    for _ in range(2000):
        a, b = rng.uniform(0.5, 360.0, 2)
        v0 = rng.uniform(-720.0, 720.0)
        sign = rng.choice([-1.0, 1.0])
        values = [v0, v0 - sign * a, v0 - sign * a + sign * b]
        expected = a / 3 < b < 3 * a
        assert (g.closure_ratios(np.radians(values)) < 1).tolist() == [expected]
        assert own_side_of_centre(*values) == expected


def test_fewer_than_three_tps_have_no_cycle():
    assert len(g.closure_ratios(np.radians([0.0, 50.0]))) == 0


# ------------------------------------------------------------------ runs


def test_1864_circulation_with_one_small_reversal_is_not_an_episode():
    episode = classify(turning_points(VALUES_1864, times=TIMES_1864))
    # the averaged bands overlap (the reason it was accepted before) ...
    assert episode.bands_overlap
    # ... but neither of its two cycles closes
    assert episode.closed_cycles == 0
    # the better cycle: advance |-2149.59 + 1922.14| = 227.45 over (5.30 + 232.75) / 2 = 119.03
    assert episode.closure_ratio_min == pytest.approx(227.45 / 119.025, abs=1e-3)
    assert not episode.accepted
    assert episode.reject_reason == g.RejectReason.NO_CLOSED_CYCLE
    assert episode.shape == g.EpisodeShape.NONE


def test_growing_amplitude_libration_stays_stationary():
    # centre 180 deg, half-amplitude growing 20 -> 120 deg over 21 TPs: successive half-swings within 1.3x
    amps = np.linspace(20.0, 120.0, 21)
    values = 180.0 + amps * (-1.0) ** np.arange(21)
    episode = classify(turning_points(values))
    assert episode.accepted and episode.shape == g.EpisodeShape.STATIONARY
    assert episode.closed_cycles == 19


def test_capture_with_an_open_entry_cycle_is_accepted():
    # entry swing of 330 deg, then a libration of half-swings 100 deg about -50: the first cycle (330 out, 100 back) is
    # open, the other 8 close
    values = [230.0] + [-100.0, 0.0] * 5
    episode = classify(turning_points(values))
    assert (g.closure_ratios(np.radians(values)) < 1).tolist() == [False] + [True] * 8
    assert episode.closed_cycles == 8
    assert episode.accepted and episode.shape == g.EpisodeShape.STATIONARY


def test_tadpole_to_horseshoe_keeps_the_episode():
    # a tadpole about 60 deg (half-amplitude 30) turning into a horseshoe 20..340 deg: the transition cycle is open
    tadpole = [90.0, 30.0, 90.0, 30.0, 90.0, 30.0]
    horseshoe = [340.0, 20.0, 340.0, 20.0, 340.0, 20.0, 340.0]
    # only (90, 30, 340) is open: 60 out, 310 back; (30, 340, 20) is 310 out, 320 back
    assert (g.closure_ratios(np.radians(tadpole + horseshoe)) < 1).tolist() == [True] * 4 + [False] + [True] * 6
    episode = classify(turning_points(tadpole + horseshoe))
    assert episode.accepted and episode.closed_cycles == 10


def test_modulated_centre_with_closed_cycles_is_accepted():
    # the centre swings +-70 deg with 8 libration periods; half-amplitude 60 deg: the centre moves by at most
    # 70 x 2 pi / 16 = 27 deg per half-period, the half-swings stay within (120 - 27, 120 + 27)
    j = np.arange(65)
    values = 70.0 * np.sin(2 * np.pi * j / 16) + 60.0 * (-1.0) ** j
    episode = classify(turning_points(values))
    assert episode.closed_cycles == 63
    assert episode.accepted and episode.shape == g.EpisodeShape.MODULATED


def test_drift_keeps_centre_moves_for_the_drift_rule():
    # s_j = 40 j + 10 (-1)^j: half-swings 60 and 20, no cycle closes, and the centre moves (10 x 40 > 2 x 10): the run
    # stays `centre_moves`, which the drift rule (G5) counts
    values = 40.0 * np.arange(13) + 10.0 * (-1.0) ** np.arange(13)
    tp = turning_points(values)
    episode = classify(tp)
    assert episode.closed_cycles == 0
    assert episode.reject_reason == g.RejectReason.CENTRE_MOVES
    assert g.drift_fraction([episode], (0.0, 1200.0), S) == pytest.approx(1.0)


# ------------------------------------------------------------------ the real record


@pytest.fixture(scope='module')
def analysis_1864():
    if not io.example_path(EXAMPLE_1864).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    return analyse_mmr(io.load_series(EXAMPLE_1864), S)


def test_1864_reads_circulation_again(analysis_1864):
    """Since W16 each run of provisional TPs has its own prominence: the stretch of the 5-deg reversal (15.4-16.5 kyr)
    has one above that swing, so its two TPs are not selected at all (before, the regime's 3.7 deg kept them and the
    closure rule rejected the run: `test_1864_circulation_with_one_small_reversal_is_not_an_episode` keeps that case)."""
    times = np.concatenate([run.tp_times_yr for run in analysis_1864.geometry.runs])
    assert not np.any(np.abs(times[:, None] - np.array(TIMES_1864[1:3])) < 1.0)
    assert not analysis_1864.geometry.episodes
    verdict = analysis_1864.verdict
    assert (str(verdict.status), str(verdict.angle_status)) == ('non_resonant', 'circulation')


def test_every_strip_reason_has_a_letter_in_both_legends():
    """Panel (a) draws rejected runs as a grey strip with a letter: every reason a run keeps as a strip (all but `edge`,
    a half-cycle trimmed off another run, and none) has one, and both legends (MMR and secular/ZLK) explain it."""
    from resonances.plotting.mmr_combined import LEGEND, REJECT_LETTERS
    from resonances.plotting.secular_plots import LONG_PERIOD_LEGEND

    strip_reasons = set(g.RejectReason) - {g.RejectReason.EDGE, g.RejectReason.NONE}
    assert set(REJECT_LETTERS) == strip_reasons
    assert REJECT_LETTERS[g.RejectReason.NO_CLOSED_CYCLE] == 'O'
    assert len(set(REJECT_LETTERS.values())) == len(REJECT_LETTERS)
    for legend in (' '.join(LEGEND), ' '.join(LONG_PERIOD_LEGEND)):
        assert 'O no closed cycle' in legend
        assert all(f'{letter} ' in legend for letter in REJECT_LETTERS.values())
