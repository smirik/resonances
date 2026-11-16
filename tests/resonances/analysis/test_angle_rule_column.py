"""The column geo_angle_rule: the angle decision's own rule (`AngleDecision.rule`), saved beside the status leaf rec_rule.

Every insufficient_time angle ends at the leaf G6 (`decide.decide_status`), whatever rule set its word: G5a (the slow
centre turns back once within a revolution), G5w (a slow swing in the well) or G6 itself (few_cycles, arc, stall). The
figure used to rebuild the angle rule from the ledger and named G5a on few_cycles rows (1992 SZ 1M-10J+1, 2002 BG
1E-10S-1: B6b's ripple test writes the G5a entry too); the column says it directly.

Synthetic records near 2J-1 (`test_policy_end_to_end.series`, 1-yr step over 12 kyr, a constant, so the identity of
any ripple fails), the expected rule worked out from `decide._g4_to_g6` by hand:
- an arc: sigma = 0.6 rev t / T, no turning point, its range 2 pi 0.6 = 3.77 rad >= pi -> G6 arc;
- a stall: 0.3 rev, 1.88 rad < pi -> G6 stall;
- a slow swing 270 deg sin(pi t / 16 kyr) (one interior turn at 8 kyr, 4.71 rad < 2 pi) with a 3-deg ripple of 680 yr
  around the turn: the ripple is set aside (B6b), no turning point outside it, the slow centre turns back once -> G5a
  arc;
- the same with a 30-deg dip at 3 kyr (two turning points outside the ripple) -> G6 few_cycles, while the ledger holds
  the G5a entry written by the ripple test (the trap the figure fell into).
"""

import math

import numpy as np
import pytest

from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.decide import G5A_ONE_REVOLUTION
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype
from tests.resonances.analysis.test_policy_end_to_end import N, T, series

S = AnalysisSettings()


def slow_swing(dip_deg: float = 0.0) -> np.ndarray:
    centre = math.radians(270.0) * np.sin(np.pi * T / 16e3)
    ripple = math.radians(3.0) * np.exp(-0.5 * ((T - 8e3) / 1400.0) ** 8) * np.sin(2 * np.pi * T / 680.0)
    return centre + ripple - math.radians(dip_deg) * np.exp(-0.5 * ((T - 3e3) / 250.0) ** 2)


def analysed(sigma, rate=None):
    rate = np.zeros(N) if rate is None else rate
    return analyse_mmr(series(sigma, rate), S, scan_owner=False, scan_subresonances=False)


RECORDS = {
    'arc': (lambda: (2 * np.pi * 0.6 * T / T[-1], np.full(N, 2 * np.pi * 0.6 / T[-1])), AngleSubtype.ARC, 'G6'),
    'stall': (lambda: (2 * np.pi * 0.3 * T / T[-1], np.full(N, 2 * np.pi * 0.3 / T[-1])), AngleSubtype.STALL, 'G6'),
    'slow arc': (lambda: (slow_swing(), None), AngleSubtype.ARC, 'G5a'),
    'ripple and few cycles': (lambda: (slow_swing(dip_deg=30.0), None), AngleSubtype.FEW_CYCLES, 'G6'),
}


@pytest.mark.parametrize('name', list(RECORDS))
def test_an_insufficient_time_row_saves_its_angle_rule(name):
    record, subtype, rule = RECORDS[name]
    analysis = analysed(*record())
    v = analysis.verdict
    assert (v.angle_status, v.angle_subtype) == (AngleStatus.INSUFFICIENT_TIME, subtype)
    numbers = analysis.numbers_row()
    assert numbers['rec_rule'] == 'G6'  # the status leaf, unchanged
    assert numbers['geo_angle_rule'] == rule


def test_the_g5a_entry_alone_does_not_make_a_g5a_row():
    """1992 SZ / 2002 BG pattern: the ripple test held G5a's one-revolution comparison, the angle went to G6."""
    analysis = analysed(slow_swing(dip_deg=30.0))
    held = [entry for entry in analysis.verdict.decision_path if entry.name == G5A_ONE_REVOLUTION and entry.result]
    assert held and held[0].value == pytest.approx(math.radians(270.0), abs=0.2)  # 4.71 rad < 2 pi
    assert analysis.numbers_row()['geo_angle_rule'] == 'G6'


def test_a_libration_saves_g2():
    # 12 cycles of 1000 yr: an episode covering the record (G2), leaf 5d/6 or another, never G6
    w = 2 * np.pi / 1000.0
    numbers = analysed(np.cos(w * T), -w * np.sin(w * T)).numbers_row()
    assert numbers['geo_angle_rule'] == 'G2' and numbers['rec_rule'] != 'G6'
