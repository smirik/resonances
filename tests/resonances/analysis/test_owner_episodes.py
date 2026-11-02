"""A1 (W28, cards g04-0..g04-3): the owner per episode.

- The owner is credited only with its measured regimes: no slip bridge between two libration pieces (g04-2).
- F7 is checked on every accepted episode; the covered ones that pass P2 are induced, the others are the row's own and
  decide the status (g04-0, g04-3).
- P2: an owner in its libration regime makes sigma's cycles only if Psi/u turns less than once per cycle.
- The second capture keeps its test window: the hull of the owner intervals that meet sigma's episodes.

The expected numbers come from the constructions or from the benchmark dumps of W28 (Claude's and Codex's g04 scratch:
2000 SF8 owner pieces 1905-5236 and 13558-16884 yr, episode 8352-11593 yr, own_psi_rate 21.618 rad/kyr, P 925.2 yr;
78085 2E-7 pieces and the old slip-joined block 3441-81604 yr), not from the code under test.
"""

import math

import pytest

from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import analyse, decide, owner
from resonances.resonance.analysis.mmr.decide import AngleDecision
from resonances.resonance.analysis.mmr.models import EpisodeCover, EpisodePhysics, OwnerCandidate, OwnerResult, ResonanceSpec
from resonances.resonance.analysis.pendulum import EnergyAnalysis
from resonances.resonance.analysis.rules import Rule
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, CheckValue, Flag, LibrationOrigin, Reason, Status
from tests.resonances.analysis.builders import make, make_episode, make_geometry, make_response, make_sampling

S = settings_from_config()
NAN = float('nan')
EPISODIC = AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, 'G2', 1)
CIRCULATION = AngleDecision(AngleStatus.CIRCULATION, AngleSubtype.NONE, 'G4')
SAMPLING = make_sampling(
    unwrap_resid_p99_rad=0.0, unwrap_resid_max_rad=0.0, unwrap_ok=True, undefined_phase=False, verdict='resolved', alias_numpy_differs=False
)


def _regime(kind, starts):
    return owner.RegimeFit(kind, tuple(starts), 0.9, (0.9,) * len(starts), 0.9, 1.0, 0.0, 0.0, 0.0)


def _accepted(regime=owner.REGIME_LIBRATION, route='r2', key='17E-36+0+19'):
    return OwnerCandidate(
        spec=ResonanceSpec.from_key(key),
        rate_rad_yr=0.0,
        r2_median=0.96,
        r2_windows=(0.96,) * 10,
        phi0_coherence=0.99,
        psi_share=0.0,
        force_au_yr=1.0,
        force_theory_au_yr=1.0,
        strength_ratio=0.005,
        rank=1,
        accepted=True,
        reason=f'accepted:{route}',
        regime=regime,
        route=route,
    )


def _owner(intervals, windows, psi_rate_rad_yr=0.0, regime=owner.REGIME_LIBRATION, decisive=None, induced_ids=frozenset()):
    """An accepted owner with its intervals, measured on the episode `windows` (id, t0, t1, period) as the analysis
    does (`owner.cover_episodes`); the decisive episode is `decisive` (default the last window)."""
    candidate = _accepted(regime)
    detail = 'owner_libration' if regime == owner.REGIME_LIBRATION else 'owner_forced_wave'
    result = make(
        OwnerResult,
        n_candidates=1,
        candidates=[candidate],
        accepted=candidate,
        check=CheckValue.FAIL,
        psi_rate_rad_yr=psi_rate_rad_yr,
        detail=detail,
        intervals=list(intervals),
        induced_ids=frozenset(induced_ids),
    )
    _, t0, t1, period = windows[-1] if decisive is None else decisive
    result.decisive_window = (t0, t1)
    result.cover = owner.covered_fraction(result.intervals, (t0, t1))
    owner.cover_episodes(result, windows, period)
    return result


# --------------------------------------------------------------------------------------------- intervals (g04-2)


def test_no_slip_piece_across_a_fifty_turn_gap():
    # 2000 SF8: 17E-36 librates 1905-5236 and 13558-16884 yr and circulates ~50 turns between; sigma librates 8352-11593
    windows, length = 6, 1000.0  # overlapping windows: each libration stretch is one piece

    def starts(t0, t1):
        return [t0 + i * (t1 - t0 - length) / (windows - 1) for i in range(windows)]

    librating = _regime(owner.REGIME_LIBRATION, starts(1905.0, 5236.0) + starts(13558.0, 16884.0))
    intervals = owner.owner_intervals([librating], length)
    assert [kind for _, _, kind in intervals] == ['libration', 'libration']
    assert intervals[0] == pytest.approx((1905.0, 5236.0, 'libration')) and intervals[1] == pytest.approx((13558.0, 16884.0, 'libration'))
    assert owner.covered_fraction(intervals, (8352.0, 11593.0)) == 0.0  # it was 1.0 with the slip 5236-13558


def test_owner_per_episode_fires_only_where_the_bridge_mattered():
    # SF8-like: librations before and after sigma's episode; the old bridge would have covered it all
    intervals = [(1905.0, 5236.0, 'libration'), (13558.0, 16884.0, 'libration')]
    assert owner.bridged_cover(intervals, (8352.0, 11593.0)) == 1.0
    sf8 = _owner(intervals, [(0, 8352.0, 11593.0, 925.226)])
    kept, flags = decide.owner_in_episode(sf8, S, Ledger.from_settings(S))
    assert kept is None and flags == [str(Flag.OWNER_OUTSIDE_EPISODE)]
    assert decide.owner_rules(sf8, flags, S) == frozenset({Rule.OWNER_PER_EPISODE})
    # 1627-like: the owner librates only after sigma's episode, with or without the bridge (cover 0 either way)
    after = _owner([(20000.0, 30000.0, 'libration')], [(0, 2000.0, 5000.0, 250.0)])
    assert owner.bridged_cover(after.intervals, after.decisive_window) == 0.0
    kept, flags = decide.owner_in_episode(after, S, Ledger.from_settings(S))
    assert kept is None and flags == [str(Flag.OWNER_OUTSIDE_EPISODE)]
    assert decide.owner_rules(after, flags, S) == frozenset()
    # a forced wave between two librations is part of the union already; the bridge adds only the uncovered gaps
    mixed = [(0.0, 200.0, 'libration'), (250.0, 350.0, 'forced_wave'), (400.0, 550.0, 'libration')]
    assert owner.covered_fraction(mixed, (100.0, 500.0)) == pytest.approx(300.0 / 400.0)
    assert owner.bridged_cover(mixed, (100.0, 500.0)) == pytest.approx(1.0)


def test_the_capture_window_is_the_old_block():
    # 78085 5J-4S-1: the 2E-7 libration pieces and sigma's seven episodes (Codex's slip-impact dump); the slip-joined
    # block the second capture was tested on before A1 was 3441-81604 yr
    pieces = [(3441, 21479), (27491, 33504), (36510, 45529), (48535, 54548), (57554, 66573), (72585, 81604)]
    intervals = [(float(a), float(b), 'libration') for a, b in pieces]
    episodes = [(1552, 11026), (14591, 21715), (26921, 32549), (36722, 45795), (50212, 53986), (59256, 66808), (72442, 79264)]
    assert owner.capture_spans(intervals, episodes) == [(3441.0, 81604.0)]
    # the decisive episode alone would see only its own piece: the capture window is not the owner's time
    assert owner.capture_spans(intervals, [(36722, 45795)]) == [(36510.0, 45529.0)]
    assert owner.capture_spans(intervals, [(85000, 90000)]) == []
    # its cover of the decisive episode without the slips (Claude's episode_cover.csv: 0.971)
    assert owner.covered_fraction(intervals, (36722.0, 45795.0)) == pytest.approx((45529 - 36722) / (45795 - 36722))
    assert owner.covered_fraction(intervals, (36722.0, 45795.0)) == pytest.approx(0.971, abs=0.001)


# --------------------------------------------------------------------------------------------- P2


def test_p2_turns_by_hand():
    # 2000 SF8 8M-9: |own_psi_rate| 21.618 rad/kyr x P 925.226 yr / 2 pi = 3.183 turns per cycle: too fast
    sf8 = _owner([(1905.0, 5236.0, 'libration')], [(0, 1905.0, 5236.0, 925.226)], psi_rate_rad_yr=0.021618)
    assert owner.psi_turns(sf8, 925.226) == pytest.approx(3.183, abs=0.001)
    assert sf8.episode_covers[0].psi_turns == pytest.approx(3.183, abs=0.001)
    assert not sf8.episode_covers[0].induced(S.mmr_owner_cover_min, S.mmr_owner_psi_turns_per_cycle_max)
    # 2013 TW135 8M-4S-7 (owner 10E-17): 0.201 rad/kyr x 780.19 yr / 2 pi = 0.025: induced
    tw135 = _owner([(0.0, 20000.0, 'libration')], [(0, 5000.0, 9000.0, 780.19)], psi_rate_rad_yr=-0.000201)
    assert tw135.psi_turns_per_cycle == pytest.approx(0.02496, abs=1e-4)
    assert tw135.episode_covers[0].induced(S.mmr_owner_cover_min, S.mmr_owner_psi_turns_per_cycle_max)
    # a forced-wave owner and the route 'identity' are not tested (nan), whatever the psi rate
    forced = _owner([(0.0, 20000.0, 'forced_wave')], [(0, 5000.0, 9000.0, 780.19)], 1.0, regime=owner.REGIME_FORCED_POSITIVE)
    assert math.isnan(forced.psi_turns_per_cycle) and math.isnan(forced.episode_covers[0].psi_turns)
    forced.accepted = _accepted(route=owner.GAIN_IDENTITY)
    assert math.isnan(owner.psi_turns(forced, 780.19))


def test_p2_refuses_a_covering_owner_whose_psi_turns_too_fast():
    windows = [(0, 1000.0, 3000.0, 925.226)]
    fast = _owner([(0.0, 5000.0, 'libration')], windows, psi_rate_rad_yr=0.021618)  # cover 1, 3.18 turns
    ledger = Ledger.from_settings(S)
    kept, flags = decide.owner_in_episode(fast, S, ledger)
    assert kept is None and flags == [str(Flag.OWNER_PSI_TOO_FAST)]
    entry = [e for e in ledger if e.name == decide.OWNER_PSI_TURNS][0]
    assert entry.result is False and entry.column == 'own_psi_turns_per_cycle' and entry.threshold == S.mmr_owner_psi_turns_per_cycle_max
    # P2 refused it, not the cover: owner_psi_turns alone
    assert decide.owner_rules(fast, flags, S) == frozenset({Rule.OWNER_PSI_TURNS})
    slow = _owner([(0.0, 5000.0, 'libration')], windows, psi_rate_rad_yr=0.00017)  # 0.025 turns
    kept, flags = decide.owner_in_episode(slow, S, Ledger.from_settings(S))
    assert kept is slow and flags == [] and decide.owner_rules(slow, flags, S) == frozenset()


# --------------------------------------------------------------------------------------------- the status (g04-0, g04-3)


def _physics(*ids):
    return [make(EpisodePhysics, episode_id=i, a_band_std_before_au=0.1, a_band_std_in_au=1.0) for i in ids]


def _decide(owner_result, geometry, angle=EPISODIC, ids=(0, 1)):
    """decide_status of an episodic angle whose own force law and potential pass (the 5d/6 dynamic leaf without an
    owner, as in test_w18_branch_rules)."""
    response = make_response(
        a_response=CheckValue.NOT_MEASURED,
        a_amplitude=CheckValue.PASS,
        slow_libration=CheckValue.NOT_APPLICABLE,
        identity=CheckValue.PASS,
        b1_ratio=0.9,
        force_r2_h1=NAN,
        force_r2_h2=NAN,
        force_contaminated=False,
    )
    energy = EnergyAnalysis(None, None, 1.0, 1.0, with_slow=CheckValue.PASS)
    ledger = Ledger.from_settings(S)
    decision = decide.decide_status(
        angle, SAMPLING, geometry, energy, response, None, None, owner_result, None, S, ledger, episodes=_physics(*ids)
    )
    return decision


def _two_episodes(induced_ids=frozenset()):
    runs = [
        make_episode(episode_id=0, tp_times_yr=[1000.0, 1500.0, 2000.0, 2500.0], period_med_yr=1000.0, centre_estimates=2),
        make_episode(episode_id=1, tp_times_yr=[7000.0, 7500.0, 8000.0, 8500.0], period_med_yr=1000.0, centre_estimates=2),
    ]
    return make_geometry(window=(0.0, 9000.0), runs=runs, induced_ids=frozenset(induced_ids))


WINDOWS = [(0, 1000.0, 2500.0, 1000.0), (1, 7000.0, 8500.0, 1000.0)]


def test_owner_before_and_after_the_episode_takes_nothing():
    # the owner librates on [0, T/3] and [2T/3, T], sigma on [T/3, 2T/3]: cover 0, the row's own transient
    t = 9000.0
    result = _owner([(0.0, t / 3, 'libration'), (2 * t / 3, t, 'libration')], [(1, t / 3, 2 * t / 3, 1000.0)])
    assert result.cover == 0.0
    geometry = make_geometry(
        window=(0.0, t), runs=[make_episode(episode_id=1, tp_times_yr=[3000.0, 3500.0, 4000.0, 4500.0, 5000.0, 5500.0, 6000.0])]
    )
    decision = _decide(result, geometry, ids=(1,))
    assert decision.status == Status.TRANSIENT and decision.origin != LibrationOrigin.INDUCED
    assert decision.reason != Reason.INDUCED_LIBRATION and str(Flag.OWNER_OUTSIDE_EPISODE) in decision.flags
    assert decision.owner == '17E-36+0+19' and Rule.OWNER_PER_EPISODE in decision.rules_fired


def test_one_induced_and_one_own_episode_is_transient():
    # the owner librates over E0 only: E0 induced, E1 the row's own
    result = _owner([(500.0, 3000.0, 'libration')], WINDOWS)
    induced, own = owner.induced_split(result, S)
    assert [e.episode_id for e in induced] == [0] and [e.episode_id for e in own] == [1]
    assert analyse._mixed_owner(result, S) == frozenset({0})  # the analysis runs once more without E0
    # the second pass: E0 out of the geometry, the owner read on the own episode E1
    geometry = _two_episodes({0})
    assert [run.episode_id for run in geometry.episodes] == [1]
    second = _owner([(500.0, 3000.0, 'libration')], WINDOWS, induced_ids={0})
    assert owner.induced_split(second, S) == ([], [second.episode_covers[1]])
    decision = _decide(second, geometry)
    assert decision.status == Status.TRANSIENT and decision.origin == LibrationOrigin.DYNAMIC
    assert str(Flag.OWNER_OUTSIDE_EPISODE) in decision.flags and decision.owner == '17E-36+0+19'
    assert Rule.OWNER_PER_EPISODE in decision.rules_fired
    # episodes.csv: E0 names its owner, E1 keeps its own origin
    episodes = analyse._induced_origins(_physics(0, 1), decision, second, S)
    assert [(p.libration_origin, p.owner) for p in episodes] == [('induced', '17E-36+0+19'), ('', '')]


def test_every_episode_induced_is_rule_2():
    result = _owner([(500.0, 9000.0, 'libration')], WINDOWS)
    assert analyse._mixed_owner(result, S) == frozenset()
    decision = _decide(result, _two_episodes())
    assert (decision.status, decision.reason, decision.origin, decision.rule) == (
        Status.NON_RESONANT,
        Reason.INDUCED_LIBRATION,
        LibrationOrigin.INDUCED,
        '2',
    )
    assert decision.rules_fired == frozenset()
    episodes = analyse._induced_origins(_physics(0, 1), decision, result, S)
    assert [p.libration_origin for p in episodes] == ['induced', 'induced']


def test_a_circulating_driver_still_names_the_origin():
    # a forced wave of a circulating owner over the whole record: no P2, induced as before
    driver = _owner([(0.0, 9000.0, 'forced_wave')], WINDOWS, psi_rate_rad_yr=0.05, regime=owner.REGIME_FORCED_POSITIVE)
    decision = _decide(driver, _two_episodes())
    assert (decision.status, decision.reason, decision.origin) == (Status.NON_RESONANT, Reason.INDUCED_LIBRATION, LibrationOrigin.INDUCED)
    assert decision.origin_detail == 'owner_forced_wave'
    record = _owner([(0.0, 9000.0, 'forced_wave')], [(None, 0.0, 9000.0, 1000.0)], 0.05, regime=owner.REGIME_FORCED_POSITIVE)
    decision = _decide(record, make_geometry(window=(0.0, 9000.0)), angle=CIRCULATION, ids=())
    assert (decision.status, decision.origin) == (Status.NON_RESONANT, LibrationOrigin.INDUCED)


def test_episode_covers_column():
    result = _owner([(500.0, 3000.0, 'libration')], WINDOWS)
    columns = result.to_columns()
    assert columns['own_episode_covers'] == '0:1.000;1:0.000'
    assert columns['own_psi_turns_per_cycle'] == 0.0
    window = _owner([(0.0, 4500.0, 'libration')], [(None, 0.0, 9000.0, 1000.0)])
    assert window.to_columns()['own_episode_covers'] == 'window:0.500'
    assert EpisodeCover(None, 0.0, 1.0, 1.0, 0.5).induced(0.5, 1.0)


# --------------------------------------------------------------------------------------------- the record edges (FIX-owner-edge)

# 2001 HW7 5J+6S-2 without planets (W28 tfix-bench dump): 5E-16 librates in one measured piece 630.15-18981.43 yr of
# the 0-19999.9 yr record (the band-pass trim 1 x 2 x P_ref ~ 630 yr at the start, the trim and the windows' step at the
# end); sigma's episodes 0 (140.03-1120.27 yr) and 8 (19494.76-19784.84 yr) lie at the edges, 4 in the middle
HW7_PIECE = (630.153998333, 18981.434805682347)
HW7_EPISODES = [
    (0, 140.034221852, 1120.27377481, 345.084332421),
    (4, 11372.7793032, 14823.6226275, 290.070888),
    (8, 19494.7641706, 19784.8350588, 290.070888),
]


def _edge_owner(intervals, windows, measured):
    """`_owner` with the owner's measured span given to F7 per episode, as `describe_owner` does."""
    result = _owner(intervals, windows)
    owner.cover_episodes(result, windows, windows[-1][3], measured, S.mmr_owner_cover_min)
    return result


def test_measured_span_is_the_hull_of_the_frame_windows():
    # windows of 1000 yr stepping by 500 yr from 600 yr (the trim): 600-1600, ..., 4600-5600, then 5000-6000
    windows = [(600.0 + 500.0 * i, 1600.0 + 500.0 * i) for i in range(9)] + [(5000.0, 6000.0)]
    assert owner.measured_span(windows) == (600.0, 6000.0)
    assert owner.measured_span(windows[:5]) == (600.0, 3600.0)
    assert all(math.isnan(x) for x in owner.measured_span([]))


def test_a_cover_of_exactly_half_is_half():
    # HW7 episode 0: the piece starts at the episode's midpoint (140.034 + 980.240 / 2 = 630.154 yr); the plain ratio
    # was 0.49999999999796, below MMR_OWNER_COVER_MIN = 0.5 by round-off alone
    assert owner.covered_fraction([(*HW7_PIECE, 'libration')], HW7_EPISODES[0][1:3]) == 0.5
    assert owner.covered_fraction([(0.0, 1.0, 'libration')], (0.0, 3.0)) == round(1.0 / 3.0, 9)


def test_an_edge_episode_the_owner_cannot_measure_is_not_its_own():
    # by hand, measured span 600-9000, owner libration 600-9000, cover_min 0.5:
    #   (9200, 9800): covered 0, unmeasured 1 -> the edge decides -> nan (neither covers nor refuses)
    #   (100, 1100): covered 500/1000 = 0.5 -> induced on the measured part alone
    #   (100, 1000): covered 400/900 = 0.444, unmeasured 500/900 -> could reach 0.5 -> nan
    measured, piece = (600.0, 9000.0), [(600.0, 9000.0, 'libration')]
    assert math.isnan(owner.episode_cover(piece, (9200.0, 9800.0), measured, 0.5))
    assert owner.episode_cover(piece, (100.0, 1100.0), measured, 0.5) == 0.5
    assert math.isnan(owner.episode_cover(piece, (100.0, 1000.0), measured, 0.5))
    assert owner.episode_cover(piece, (9200.0, 9800.0)) == 0.0  # without the span: the rule before
    # an owner refused where it is measured: libration 5000-9000, measured 600-9000 (a refused regime before)
    #   (300, 1500): covered 0, unmeasured 300/1200 = 0.25 -> at most 0.25 < 0.5 -> refused, cover 0
    #   (100, 1100): covered 0, unmeasured 0.5 -> could reach 0.5 -> nan
    late = [(5000.0, 9000.0, 'libration')]
    assert owner.episode_cover(late, (300.0, 1500.0), measured, 0.5) == 0.0
    assert math.isnan(owner.episode_cover(late, (100.0, 1100.0), measured, 0.5))
    assert owner.unmeasured_fraction((300.0, 1500.0), measured) == 0.25
    assert owner.unmeasured_fraction((300.0, 1500.0), (NAN, NAN)) == 0.0


def test_hw7_edge_episodes_leave_the_row_induced():
    # with the measured span: 0 induced (0.5), 4 induced (1.0), 8 not measured -> no own episode, rule 2 (the row
    # with the planets: non_resonant / induced_libration); without it 8 had cover 0 and made the mixed path
    intervals = [(*HW7_PIECE, 'libration')]
    result = _edge_owner(intervals, HW7_EPISODES, HW7_PIECE)
    covers = [entry.cover for entry in result.episode_covers]
    assert covers[:2] == [0.5, 1.0] and math.isnan(covers[2])
    assert result.to_columns()['own_episode_covers'] == '0:0.500;4:1.000'
    induced, own = owner.induced_split(result, S)
    assert [e.episode_id for e in induced] == [0, 4] and own == []
    assert analyse._mixed_owner(result, S) == frozenset()
    kept, flags = decide.owner_in_episode(result, S, Ledger.from_settings(S))
    assert kept is result and flags == []
    before = _owner(intervals, HW7_EPISODES)
    assert [e.episode_id for e in owner.induced_split(before, S)[1]] == [8]
    assert analyse._mixed_owner(before, S) == frozenset({0, 4})


def test_an_owner_absent_where_measured_still_refuses_beside_an_edge_episode():
    # 2000 SF8-like: the owner librates 1905-5236 yr only, measured 600-19400 yr; sigma's episode 8352-11593 yr is
    # inside the measured span and uncovered (own), the edge episode 19500-19800 yr is not measured: own wins
    windows = [(0, 8352.0, 11593.0, 925.226), (1, 19500.0, 19800.0, 925.226)]
    result = _edge_owner([(1905.0, 5236.0, 'libration')], windows, (600.0, 19400.0))
    induced, own = owner.induced_split(result, S)
    assert induced == [] and [e.episode_id for e in own] == [0]
    kept, flags = decide.owner_in_episode(result, S, Ledger.from_settings(S))
    assert kept is None and flags == [str(Flag.OWNER_OUTSIDE_EPISODE)]


# --------------------------------------------------------------------------------------------- benchmark (local data)

# (example, (status, angle_status, confidence, libration_origin), own_cover): the covers without the slip bridge were
# measured on the benchmark before A1 by removing the slip pieces by hand (W28 g04 measurements: 1685 0.741, GK3
# 0.789/0.913, 78085 0.971, 88710_c7 0.454, 1992 SZ 0.418); the statuses are the author's labels or the previous run's
OWNER_ROWS = [
    ('mmr2__2000_SF8__8M-9+0+1', ('transient', 'episodic', 'medium', 'dynamic'), 0.0),
    ('mmr3__78085__5J-4S-1+0+0+0', ('non_resonant', 'episodic', 'medium', 'induced'), 0.971),
    ('mmr3__1685__5J+6S-1+0+0-10', ('non_resonant', 'episodic', 'medium', 'induced'), 0.741),
    ('mmr3__1997_GK3__3M+5S-3+0+0-5', ('non_resonant', 'episodic', 'medium', 'induced'), 0.789),
    ('mmr3__1997_GK3__3M+5S-3+0+0-5__rerun', ('non_resonant', 'drift', 'medium', 'induced'), 0.913),
    ('mmr2__88710_c7__7V-11+0+4', ('non_resonant', 'circulation', 'certain', ''), 0.454),
    ('mmr3__1992_SZ__1M-10J+1+0+0+8__100kyr', ('non_resonant', 'circulation', 'medium', ''), 0.418),
]


@pytest.mark.benchmark
@pytest.mark.parametrize('example, expected, cover', OWNER_ROWS)
def test_benchmark_owner_per_episode(example, expected, cover):
    """SF8: 17E-36 circulates ~50 turns between its two librations, so without the bridge it covers none of sigma's
    episode (and its Psi/u turns 3.18 times per cycle): the row's own transient (author 2026-10-31, card g04-1). The
    other owners keep their rows; 88710_c7 and 1992 SZ lose the origin induced (cover below 0.5)."""
    from resonances.resonance.analysis.mmr import io
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    analysis = analyse_mmr(io.load_series(example), S)
    v, numbers = analysis.verdict, analysis.numbers_row()
    assert (str(v.status), str(v.angle_status), str(v.confidence), str(v.libration_origin)) == expected
    assert numbers['own_cover'] == pytest.approx(cover, abs=0.02)
    assert ':slip' not in numbers['own_intervals']
    if expected[3] == '':
        assert str(Flag.OWNER_OUTSIDE_EPISODE) in v.flags and Rule.OWNER_PER_EPISODE in v.rules_fired
    if example.startswith('mmr2__2000_SF8'):
        assert numbers['own_psi_turns_per_cycle'] == pytest.approx(3.18, abs=0.02)
        assert [p['episode_id'] for p in analysis.episode_rows() if p['accepted']] == [0]
    if example.startswith('mmr3__78085'):
        assert numbers['own_psi_sweep_deg'] == pytest.approx(155.0, abs=5.0)  # the second capture's window is kept
        # A2 (card g05-6 C): excluded by route F3, x_ddot follows the 2 omega cycle; the sweep alone does not exclude
        assert numbers['own_second_capture'] == 'excluded' and numbers['own_second_capture_route'] == 'secular_forcing'
