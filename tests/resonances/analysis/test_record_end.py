"""How a record's edges end (W28 C14, rule end_arc_extend; cards g01-7 A, g01-4 A, g02-4 A, q-4 A).

The measurement (`geometry.edge_stretch`) is checked on piecewise-linear sigma_f whose numbers are worked out by hand:
a circulation of 5 revolutions in 250 yr (one per 50 yr) that then slows to a 90 deg arc over 500 yr ends in a
slowdown 500 yr long after a 50-yr revolution; the same circulation to the end leaves less than one revolution's time;
a backward lap of 90 deg over 150 yr before a centre turn and 6 revolutions after it starts in one reflection. The
decision (`decide.record_edges`) is checked on real geometries (`builders`) holding those stretches."""

import math
import time

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.confidence import CAP_CEILING, Cap
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import decide
from resonances.resonance.analysis.mmr.decide import StatusDecision
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import (
    AngleStatus,
    Confidence,
    Flag,
    LibrationOrigin,
    Reason,
    ResonanceType,
    Status,
    Verdict,
)
from tests.resonances.analysis.builders import make_centre, make_episode, make_geometry, make_turning_points

SETTINGS = AnalysisSettings()
STEP = 0.5  # yr
TWO_PI = 2 * np.pi


def slowing_circulation(arc_rad=np.pi / 2):
    """sigma_f: 5 revolutions in 0-250 yr (one per 50 yr), then `arc_rad` monotonically over 250-750 yr."""
    t = np.arange(0.0, 750.0 + STEP / 2, STEP)
    sigma = np.where(t <= 250.0, TWO_PI * t / 50.0, 5 * TWO_PI + arc_rad * (t - 250.0) / 500.0)
    return t, sigma


def monotone_centre(t, sigma):
    """A centre without turns: one pass over the window."""
    return make_centre(passes=[g.MonotonePass(float(t[0]), float(t[-1]), float(sigma[-1] - sigma[0]))])


def test_a_circulation_that_slows_down_ends_in_a_slowdown():
    """End: the 5th revolution completes at 250 yr (each took 50 yr), then 500 yr without another; range and travel
    90 deg. Start: going back from the end, the 5th revolution completes at 12.5 yr: the stretch before it is
    12.5 yr, shorter than any revolution."""
    t, sigma = slowing_circulation()
    window = (0.0, 750.0)
    end = g.edge_stretch(t, sigma, window, monotone_centre(t, sigma), g.LEG_EDGE_END)
    assert end.kind == g.EDGE_SLOWDOWN and end.pass_rev == pytest.approx(5.25)
    assert end.t_mark_yr == pytest.approx(250.0, abs=STEP) and end.last_rev_yr == pytest.approx(50.0, abs=STEP)
    assert end.duration_yr == pytest.approx(500.0, abs=STEP)
    arc_step = (np.pi / 2) * STEP / 500.0  # the mark may fall one sample late
    assert end.range_rad == pytest.approx(np.pi / 2, abs=arc_step) and end.travel_rad == pytest.approx(np.pi / 2, abs=arc_step)
    start = g.edge_stretch(t, sigma, window, monotone_centre(t, sigma), g.LEG_EDGE_START)
    assert start.kind == g.EDGE_SLOWDOWN and start.t_mark_yr == pytest.approx(12.5, abs=STEP)
    # its longest revolution is the first one back from the end: the 90 deg arc (500 yr) and 270 deg at 50 yr/rev
    assert start.duration_yr == pytest.approx(12.5, abs=STEP) and start.last_rev_yr == pytest.approx(537.5, abs=STEP)


def test_a_circulation_to_the_end_leaves_less_than_one_revolution():
    """15 revolutions in 750 yr at one per 50 yr: after the last completed one less than 50 yr remain."""
    t = np.arange(0.0, 750.0 + STEP / 2, STEP)
    sigma = TWO_PI * (t + 20.0) / 50.0  # the record starts 20 yr into a revolution
    end = g.edge_stretch(t, sigma, (0.0, 750.0), monotone_centre(t, sigma), g.LEG_EDGE_END)
    assert end.kind == g.EDGE_SLOWDOWN and end.duration_yr < end.last_rev_yr == pytest.approx(50.0, abs=STEP)


def test_revolutions_of_unequal_length_are_no_slowdown():
    """A first revolution of 80 yr, twelve of 50 yr to 680 yr, then 70 yr at the first one's rate to the end: the
    remainder is longer than the last revolution (50 yr) but not than the longest (80 yr), so no slowdown (2017 XF62
    20 kyr: 1.005 x its last revolution)."""
    t = np.arange(0.0, 750.0 + STEP / 2, STEP)
    sigma = np.where(
        t <= 80.0,
        TWO_PI * t / 80.0,
        np.where(t <= 680.0, TWO_PI * (1 + (t - 80.0) / 50.0), TWO_PI * (13 + (t - 680.0) / 80.0)),
    )
    end = g.edge_stretch(t, sigma, (0.0, 750.0), monotone_centre(t, sigma), g.LEG_EDGE_END)
    assert end.duration_yr == pytest.approx(70.0, abs=STEP) and end.last_rev_yr == pytest.approx(80.0, abs=STEP)
    assert end.duration_yr / end.last_rev_yr < 1.0  # below even the physical bound


def reflected_start():
    """sigma_f: -90 deg over 0-150 yr (a backward lap), the slow centre turns at 150 yr, then 6 revolutions to 750 yr."""
    t = np.arange(0.0, 750.0 + STEP / 2, STEP)
    sigma = np.where(t <= 150.0, -(np.pi / 2) * t / 150.0, -np.pi / 2 + 6 * TWO_PI * (t - 150.0) / 600.0)
    centre = make_centre(
        turns=make_turning_points([150.0], [-np.pi / 2]),
        passes=[g.MonotonePass(0.0, 150.0, -np.pi / 2), g.MonotonePass(150.0, 750.0, 6 * TWO_PI)],
    )
    return t, sigma, centre


def test_one_reflection_at_the_start_is_a_reflection_and_the_end_is_none():
    """Start: a quarter revolution between the edge and the centre turn (150 yr), 6 revolutions beyond it (2008 EZ84's
    shape). End: the 6th revolution completes at the window's end, nothing remains."""
    t, sigma, centre = reflected_start()
    start = g.edge_stretch(t, sigma, (0.0, 750.0), centre, g.LEG_EDGE_START)
    assert start.kind == g.EDGE_REFLECTION and start.pass_rev == pytest.approx(0.25)
    assert start.other_rev == pytest.approx(6.0) and start.t_mark_yr == pytest.approx(150.0)
    assert start.duration_yr == pytest.approx(150.0)
    end = g.edge_stretch(t, sigma, (0.0, 750.0), centre, g.LEG_EDGE_END)
    assert end.kind == g.EDGE_SLOWDOWN and end.duration_yr == pytest.approx(0.0, abs=STEP)


def test_a_long_circulation_is_read_in_a_fraction_of_a_second():
    """20 000 revolutions of 10 yr on a 1-yr step (200 006 samples, a high-order angle over 200 kyr): the end's last
    revolution completes at 200 000 yr, 5 yr remain, each revolution took 10 yr; the start mirrors it (the first
    revolution back from the end completes 5 yr after the start). The revolutions are found by bisection on the running
    maximum: one boolean scan per revolution took ~100 s per edge on 27103 14M+27 (W28 run)."""
    t = np.arange(0.0, 200005.0 + 0.5, 1.0)
    sigma = TWO_PI * t / 10.0
    cpu = time.process_time()
    end = g.edge_stretch(t, sigma, (0.0, 200005.0), monotone_centre(t, sigma), g.LEG_EDGE_END)
    start = g.edge_stretch(t, sigma, (0.0, 200005.0), monotone_centre(t, sigma), g.LEG_EDGE_START)
    cpu = time.process_time() - cpu
    assert end.kind == g.EDGE_SLOWDOWN and end.pass_rev == pytest.approx(20000.5)
    assert end.t_mark_yr == pytest.approx(200000.0, abs=1.0) and end.duration_yr == pytest.approx(5.0, abs=1.0)
    assert end.last_rev_yr == pytest.approx(10.0, abs=1.0)
    assert start.t_mark_yr == pytest.approx(5.0, abs=1.0) and start.last_rev_yr == pytest.approx(10.0, abs=1.0)
    assert cpu < 0.5


# --- the decision ---------------------------------------------------------------------------------------------------

FLOOR = 100.0  # yr: the slowdown above is 5 floor periods long
SPAN = 750.0


def geometry_with(*stretches, **fields):
    fields.setdefault('window', (0.0, SPAN))
    return make_geometry(
        edge_stretches=tuple(stretches), period_floor_yr=FLOOR, period_floor_setting='mmr_libration_period_min_yr', **fields
    )


def slowdown(duration=500.0, last_rev=50.0, range_rad=np.pi / 2, side=g.LEG_EDGE_END):
    mark = SPAN - duration if side == g.LEG_EDGE_END else duration
    return g.EdgeStretch(side, g.EDGE_SLOWDOWN, 5.25, mark, duration, last_rev, range_rad=range_rad, travel_rad=range_rad)


def circulation(status=Status.NON_RESONANT):
    return StatusDecision(status, Reason.CIRCULATION, LibrationOrigin.NONE, '', '')


def edges(geometry, decision=None, reported=AngleStatus.CIRCULATION, regime='', settings=SETTINGS, ledger=None):
    ledger = ledger if ledger is not None else Ledger.from_settings(settings)
    decision = decision if decision is not None else circulation()
    return decide.record_edges(geometry, decision, reported, regime, float('nan'), SPAN, settings, ledger)


def test_a_circulation_ending_in_a_slow_arc_gets_the_final_regime_the_cap_and_the_forward_flag():
    """A 90 deg arc over 500 yr after 50-yr revolutions (10 x, 2 x 5 floors): final_regime stall (range < pi, as G6)
    from 250 yr, the cap record_ends_unresolved (<= medium) on a non_resonant circulation, the flag forward asking for
    round_up(2 x 750 yr) = 2000 yr; its two comparisons are on the ledger. A 200 deg arc is an arc."""
    ledger = Ledger.from_settings(SETTINGS)
    found = edges(geometry_with(slowdown()), ledger=ledger)
    assert (found.final_regime, found.final_regime_start_yr) == ('stall', 250.0)
    assert found.caps == (str(Cap.RECORD_ENDS_UNRESOLVED),) and CAP_CEILING[Cap.RECORD_ENDS_UNRESOLVED] == Confidence.MEDIUM
    assert (found.direction, found.extend_to_years, found.sources) == (decide.EXTEND_FORWARD, 2000.0, ('end:slowdown',))
    assert [entry.name for entry in ledger if entry.name.startswith(decide.RULE_END_ARC)] == [
        'end arc end slowdown: 2 x length / floor',
        'end arc end slowdown: length / the longest revolution',
    ]
    assert edges(geometry_with(slowdown(range_rad=np.radians(200.0)))).final_regime == 'arc'
    assert edges(geometry_with(slowdown()), regime='circulation').final_regime == 'stall'  # after G1c's circulation


def test_an_uncertain_row_gets_the_final_regime_and_the_flag_but_no_cap():
    """2022 BB2 (uncertain/one_coorbital_leg, the word episodic): final_regime and the flag, no cap (it has no level),
    and nothing on the ledger: the flag changes no level."""
    ledger = Ledger.from_settings(SETTINGS)
    decision = StatusDecision(Status.UNCERTAIN, Reason.ONE_COORBITAL_LEG, LibrationOrigin.UNKNOWN, '', '')
    found = edges(geometry_with(slowdown()), decision, AngleStatus.EPISODIC, ledger=ledger)
    assert (found.final_regime, found.caps, found.direction) == ('stall', (), decide.EXTEND_FORWARD)
    assert not list(ledger)


@pytest.mark.parametrize(
    'stretch, fields, reported',
    [
        (slowdown(duration=40.0), {}, AngleStatus.CIRCULATION),  # a circulation to the end: 40 yr < its 50-yr revolution
        (slowdown(duration=150.0, last_rev=100.0), {}, AngleStatus.CIRCULATION),  # 1.5 x: a drifting rate, no slowdown
        (slowdown(duration=45.0, last_rev=20.0), {}, AngleStatus.CIRCULATION),  # 2.25 x slower, but 2 x 45 yr < one floor
        (slowdown(), {}, AngleStatus.LIBRATION),  # the word already says how the record ends
        (slowdown(), {}, AngleStatus.DRIFT),
        (slowdown(), {'runs': [make_episode(tp_times_yr=np.linspace(400.0, 700.0, 5), period_med_yr=150.0)]}, AngleStatus.CIRCULATION),
    ],
)
def test_no_slowdown_no_reading(stretch, fields, reported):
    """Nothing changes: the stretch is too short (against its revolution or the floor), the word says how the record
    ends, or a librating episode lies in the stretch (its own rules read that edge)."""
    found = edges(geometry_with(stretch, **fields), reported=reported)
    assert found == decide.RecordEdges('', float('nan')) or (found.direction, found.caps) == ('', ())


def test_one_reflection_at_the_start_gives_the_backward_flag_only():
    """2008 EZ84 (g01-4 A: non_resonant high, a negative control): the start reflection asks for a backward
    integration; no cap, no final regime. A start slowdown is read the same way (flag only); both edges -> both. A
    window that starts at a jump of a has no record start to extend; an angle that is not 1:1 has no reflection (q-5 i)."""
    reflection = g.EdgeStretch(g.LEG_EDGE_START, g.EDGE_REFLECTION, 0.25, 150.0, 150.0, other_rev=6.0)
    found = edges(geometry_with(reflection, coorbital=True))
    assert (found.direction, found.caps, found.final_regime) == (decide.EXTEND_BACKWARD, (), '')
    assert edges(geometry_with(slowdown(side=g.LEG_EDGE_START))).direction == decide.EXTEND_BACKWARD
    both = edges(geometry_with(reflection, slowdown(), coorbital=True))
    assert (both.direction, both.caps) == (decide.EXTEND_BOTH, (str(Cap.RECORD_ENDS_UNRESOLVED),))
    after_jump = edges(geometry_with(reflection, coorbital=True, flags=[str(Flag.LAST_REGIME_AFTER_JUMP)]))
    assert after_jump.direction == ''  # the window starts at a jump of a
    short = g.EdgeStretch(g.LEG_EDGE_START, g.EDGE_REFLECTION, 0.25, 40.0, 40.0, other_rev=6.0)  # 2 x 40 yr < a floor
    assert edges(geometry_with(short, coorbital=True)).direction == ''
    assert edges(geometry_with(reflection)).direction == ''  # not 1:1: q-5 i, a slow turn is no reflection


def test_a_circulation_that_reverses_on_its_own_slow_scale_is_no_slowdown():
    """37908 23J-8 (author non_resonant/high): the slow centre turns at 19.4, 59.9 and 75.6 kyr (40.5 kyr apart at
    most), and the end stall of 17.7 kyr after 6.6-kyr revolutions (2.7 x) is the approach to its next reversal: no
    slowdown. The same stall longer than every stretch between turns is one."""

    def reversing(duration):
        return g.EdgeStretch(g.LEG_EDGE_END, g.EDGE_SLOWDOWN, 1.84, SPAN - duration, duration, 6595.0, range_rad=1.0, reversal_yr=40513.0)

    assert edges(geometry_with(reversing(17749.0))).direction == ''
    assert edges(geometry_with(reversing(41000.0))).caps == (str(Cap.RECORD_ENDS_UNRESOLVED),)


def edge_leg(edge, exit_out_in=1.0):
    """A clean horseshoe leg cut by a record edge; its observed turn reflects (|ln out/in| <= 0.34) or not."""
    turn = g.LegTurn(650.0 if edge == g.LEG_EDGE_END else 100.0, 'planet', exit_out_in, 0.8)
    entry, exit_ = (turn, None) if edge == g.LEG_EDGE_END else (None, turn)
    t0, t1 = (650.0, SPAN) if edge == g.LEG_EDGE_END else (0.0, 100.0)
    return g.SingleLeg(t0, t1, edge, g.LEG_HORSESHOE, 2.0, entry, exit_, force='pass', force_scale=1.0, energy='pass')


def test_edge_legs_and_edge_half_librations_give_the_flag_of_their_edge():
    """q-4 A ('cut by the edge -> uncertain + extend'): a clean 1:1 edge leg whose turn reflects flags its edge
    (forward at the end, backward at the start); a kick (out/in 2) does not; nor does any leg next to a librating
    episode. An edge half-libration that holds its law (42946 43S-6) flags its edge too."""
    assert edges(geometry_with(single_legs=(edge_leg(g.LEG_EDGE_END),))).sources == ('leg:end',)
    assert edges(geometry_with(single_legs=(edge_leg(g.LEG_EDGE_START),))).direction == decide.EXTEND_BACKWARD
    assert edges(geometry_with(single_legs=(edge_leg(g.LEG_EDGE_END, 2.0),))).direction == ''
    episode = make_episode(tp_times_yr=np.linspace(200.0, 500.0, 5), period_med_yr=150.0)
    assert edges(geometry_with(single_legs=(edge_leg(g.LEG_EDGE_END),), runs=[episode])).direction == ''
    half = g.HalfLibration(28476.0, 32285.0, g.LEG_EDGE_END, 2.25, 56.0, 0.72, 0.62, True)
    assert edges(geometry_with(half_librations=(half,))).direction == decide.EXTEND_FORWARD
    weak = g.HalfLibration(28476.0, 32285.0, g.LEG_EDGE_END, 2.25, 56.0, 0.08, 0.06, True)
    assert edges(geometry_with(half_librations=(weak,))).direction == ''


def test_the_switch_off_leaves_the_verdict_as_it_was():
    off = AnalysisSettings(mmr_record_edge_stretch=False)
    found = edges(geometry_with(slowdown(), single_legs=(edge_leg(g.LEG_EDGE_END),)), settings=off, regime='circulation')
    assert (found.final_regime, found.caps, found.direction, found.extend_to_years) == ('circulation', (), '', None)


def test_the_flag_and_its_direction_round_trip_through_the_summary_row():
    verdict = Verdict(
        ResonanceType.MMR,
        Status.NON_RESONANT,
        AngleStatus.CIRCULATION,
        flags=[str(Flag.EXTEND_TIME)],
        extend_to_years=2000.0,
        extend_direction=decide.EXTEND_FORWARD,
        final_regime='stall',
        final_regime_start_yr=250.0,
    )
    row = verdict.to_flat_dict()
    assert (row['extend_direction'], row['extend_to_years'], row['flags']) == ('forward', 2000.0, 'extend_time')
    back = Verdict.from_flat_dict(row)
    assert (back.extend_direction, back.extend_to_years, back.final_regime) == ('forward', 2000.0, 'stall')
    assert math.isclose(back.final_regime_start_yr, 250.0)
