"""The zone flags follow the zone reading that survived (`decide.zone_withdrawn`): when the x2 filter guard empties Z or
L2 (the energy changes at 2 P_c), or the status rests on an episode before a jump of a (G1e: the zone was measured on
the regime after it), the row shows zone_not_measured and neither episodes_in_separatrix_zone (flag or cap) nor
libration_in_separatrix_zone. Before, three benchmark rows (79653 3M+3J-7, 2022 YG 1:1, 17886 36M-61) carried both."""

import pytest

from resonances.resonance.analysis import confidence as conf
from resonances.resonance.analysis.mmr import decide
from resonances.resonance.analysis.mmr.decide import AngleDecision, StatusDecision
from resonances.resonance.analysis.mmr.preprocess import DecisiveQuantities, FilterSensitivity
from resonances.resonance.analysis.pendulum import CycleClass
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, LibrationOrigin, Margin, MarginValue, Reason, Status
from tests.resonances.analysis.builders import make, make_energy, make_geometry

S = AnalysisSettings()
EPISODIC = AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, 'G2', 0)
TRANSIENT = StatusDecision(Status.TRANSIENT, Reason.DYNAMIC_EPISODE, LibrationOrigin.DYNAMIC, '', '')
IN_ZONE = str(conf.Cap.EPISODES_IN_SEPARATRIX_ZONE)
NOT_MEASURED = str(conf.Cap.ZONE_NOT_MEASURED)


def _in_zone():
    """An energy whose crossings have every cycle in the zone (what `zone.episodes_in_zone` reports)."""
    return make_energy(episodes_in_separatrix_zone=True, cycle_classes=[CycleClass.ZONE] * 3, flags=['episodes_in_separatrix_zone'])


def _m(margin):
    return MarginValue(margin, 0.5)


@pytest.mark.parametrize(
    'margins, withdrawn',
    [
        ({}, False),  # not graded (uncertain, a circulation): nothing to withdraw
        ({'Z': _m(Margin.FULL), 'L2': _m(Margin.THIN)}, False),
        ({'Z': _m(Margin.THIN)}, False),
        ({'Z': _m(Margin.NOT_MEASURED), 'L2': _m(Margin.THIN)}, True),
        ({'L2': _m(Margin.THIN)}, False),  # a libration: L2 carries the zone
        ({'L2': _m(Margin.NOT_MEASURED)}, True),
        ({'Z': _m(Margin.FULL), 'L2': _m(Margin.NOT_MEASURED)}, False),  # Z decides where it is graded
    ],
)
def test_the_margin_that_carries_the_zone_decides(margins, withdrawn):
    assert decide.zone_withdrawn(_in_zone(), margins) == withdrawn


def test_no_energy_of_the_regime_is_no_zone():
    assert decide.zone_withdrawn(None, {})
    # G1e: the status reads an episode before the jump; the tree view drops the last regime's energy
    before_jump = AngleDecision(AngleStatus.CIRCULATION, AngleSubtype.NONE, 'G1e', None, earlier_episode_id=0)
    view = decide.tree_view(before_jump, None, _in_zone())
    assert view.energy is None and decide.zone_withdrawn(view.energy, {})


@pytest.mark.parametrize('changed, in_zone_cap', [((), True), (('turning_points',), True), (('energy',), False)])
def test_the_filter_guard_takes_the_zone_cap_with_the_zone(changed, in_zone_cap):
    energy = _in_zone()
    margins = {'Z': _m(Margin.THIN), 'L2': _m(Margin.THIN)}
    quantities = make(DecisiveQuantities)
    sensitivity = FilterSensitivity(quantities, quantities, changed, decisive_episode_id=0)
    decide.apply_filter_sensitivity({}, margins, sensitivity)  # as analyse: before the caps read the margins
    caps = decide.collect_caps(decide.tree_view(EPISODIC, None, energy), TRANSIENT, make_geometry(), margins, S)
    assert (IN_ZONE in caps) == in_zone_cap
    assert (NOT_MEASURED in caps) == (not in_zone_cap)  # one or the other, never both
    assert decide.zone_withdrawn(energy, margins) == (not in_zone_cap)


def test_the_verdict_flags_lose_both_zone_flags_only_when_withdrawn():
    flags = ['filter_sensitive', 'episodes_in_separatrix_zone', 'libration_in_separatrix_zone', 'zone_edge_cycle']
    withdrawn = {'Z': _m(Margin.NOT_MEASURED)}
    assert decide.surviving_zone_flags(flags, _in_zone(), withdrawn) == ['filter_sensitive', 'zone_edge_cycle']
    assert decide.surviving_zone_flags(flags, _in_zone(), {'Z': _m(Margin.THIN)}) == flags
    assert decide.surviving_zone_flags(flags, None, {}) == ['filter_sensitive', 'zone_edge_cycle']
