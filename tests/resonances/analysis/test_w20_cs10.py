"""W20 CS10: certain for far circulations (H report of 2026-10-19, Claude + Codex, author's Q25).

L8's step rule (rotation per step < MMR_L8_STEP_FULL_DEG) exists to resolve a libration's turning points. On a
non_resonant circulation whose slow a stays >= MMR_FAR_FROM_RES_FACTOR a-scales from a* (sigma_dot = c (a - a*) keeps
its sign) there is nothing to resolve, so that part is not graded (`decide.far_circulation`). An alias is full in C5
when it is that far and its branch clearance per step exceeds MMR_C5_BRANCH_RESID_FACTOR x the largest unwrap residual.
The x2 guard's energy tolerance is relative far above the separatrix.

Expected values by hand (H report, a* from Kepler's law on the planets' mean axes, matching the package to 1e-4 au):
29744 23M-49: <a> 2.6033, a* 2.5227 au, 48.7 a-scales away; clearance 0.0547 turn = 19.7 deg > 2 x 8.84 deg = 17.7 deg.
88710_c3 49M-31: clearance 0.0694 turn = 25.0 deg < 2 x 16.5 deg = 33.1 deg; 88710_c7 32E-29: 12.7 deg < 45.3 deg.
30963 14M-25: h at P_c and 2 P_c 6879852 and 6879123, |dh| = 729 < 0.1 x 6879123; h 0.95 and 1.06 cross the separatrix.
"""

import math
from types import SimpleNamespace

import pytest

from resonances.resonance.analysis.ledger import Ledger, Margin
from resonances.resonance.analysis.mmr import decide, io
from resonances.resonance.analysis.mmr.preprocess import GuardOutcome, _compare_energy
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, LibrationOrigin, MarginClass, Reason, Status
from tests.resonances.analysis.test_impl_w3_guards_filter import quantities

S = settings_from_config()
DEG = math.pi / 180


def _sampling(rotation_deg=80.6, far=58.9, lines=(), clearance_deg=float('nan'), resid_deg=float('nan')):
    return SimpleNamespace(
        step_rotation_max_rad=rotation_deg * DEG,
        far_ratio=far,
        aliased_lines_yr=tuple(lines),
        branch_margin_turns=clearance_deg / 360.0,
        unwrap_resid_max_rad=resid_deg * DEG,
    )


def _angle(status=AngleStatus.CIRCULATION, subtype=''):
    return SimpleNamespace(angle_status=status, subtype=subtype)


def _decision(status=Status.NON_RESONANT):
    return decide.StatusDecision(status, Reason.CIRCULATION, LibrationOrigin.NONE, '', '')


def _l8(sampling, angle=None, decision=None, gain=float('nan')):
    prep = SimpleNamespace(gain_at_period_cycle=gain)
    return decide._l8(sampling, prep, Ledger.from_settings(S), S, angle, decision)


# --- L8: the step part is not graded for a far circulation -----------------------------------------------------------


def test_l8_step_part_not_graded_for_a_far_circulation():
    # 30963 14M-25: 80.6 deg per 0.5-yr step, 58.9 a-scales from a*: nothing else graded, so L8 is not applicable
    margin = _l8(_sampling(), _angle(), _decision())
    assert margin.margin == Margin.NOT_APPLICABLE and margin.margin_class == MarginClass.SUPPORT
    assert margin.value == pytest.approx(80.6)
    # no commensurability at any a > 0 (16397 29M+50: far_ratio inf), an alias turning 5372 deg per step
    assert _l8(_sampling(5372.0, math.inf), _angle(subtype=AngleSubtype.ALIAS), _decision()).margin == Margin.NOT_APPLICABLE


@pytest.mark.parametrize(
    'sampling, angle, status',
    [
        (_sampling(far=8.8), _angle(), Status.NON_RESONANT),  # 39423 23J-10: 8.8 a-scales, below 10
        (_sampling(far=float('nan')), _angle(), Status.NON_RESONANT),  # a* unknown: not far
        (_sampling(), _angle(AngleStatus.EPISODIC), Status.NON_RESONANT),  # not a circulation
        (_sampling(), _angle(AngleStatus.LIBRATION), Status.RESONANT),  # never a libration
    ],
)
def test_l8_step_part_still_graded(sampling, angle, status):
    assert _l8(sampling, angle, _decision(status)).margin == Margin.THIN  # 80.6 deg >= 10 deg


def test_l8_other_parts_stay_for_a_far_circulation():
    assert _l8(_sampling(lines=(206.0,)), _angle(), _decision()).margin == Margin.THIN  # an aliased line in the band
    assert _l8(_sampling(), _angle(), _decision(), gain=0.99).margin == Margin.FULL  # 0.99 >= 0.94
    assert _l8(_sampling(), _angle(), _decision(), gain=0.3).margin == Margin.FAIL  # 0.3 < 0.5
    assert _l8(_sampling(rotation_deg=5.0, far=8.8), _angle(), _decision()).margin == Margin.FULL  # 5 deg < 10 deg


def test_guard_without_comparison_thins_a_not_applicable_l8():
    margins = {'L8': _l8(_sampling(), _angle(), _decision())}
    checks = {'check_mmr_filter': 'pass'}
    decide.apply_filter_sensitivity(checks, margins, SimpleNamespace(sensitive=None))
    assert margins['L8'].margin == Margin.THIN and checks['check_mmr_filter'] == 'not_measured'


# --- C5 for an alias ------------------------------------------------------------------------------------------------


def _c5(sampling):
    return decide._c5_margin(_angle(subtype=AngleSubtype.ALIAS), None, sampling, Ledger.from_settings(S), S)


@pytest.mark.parametrize(
    'far, clearance, resid, margin',
    [
        (48.7, 19.7, 8.84, Margin.FULL),  # 29744 23M-49: 19.7 > 17.7 deg
        (93.4, 21.7, 5.52, Margin.FULL),  # 44662 54M-89: 21.7 > 11.0 deg
        (10.51, 24.97, 16.53, Margin.THIN),  # 88710_c3 49M-31: 25.0 < 33.1 deg
        (39.6, 12.66, 22.67, Margin.THIN),  # 88710_c7 32E-29: 12.7 < 45.3 deg
        (8.8, 100.0, 1.0, Margin.THIN),  # wide clearance, but not far from a*
        (48.7, 19.7, float('nan'), Margin.THIN),  # the residual not measured
    ],
)
def test_c5_alias_branch_against_the_unwrap_residual(far, clearance, resid, margin):
    assert _c5(_sampling(far=far, clearance_deg=clearance, resid_deg=resid)).margin == margin


def test_branch_clearance_in_degrees():
    assert decide.branch_clearance_deg(SimpleNamespace(branch_margin_turns=0.0547)) == pytest.approx(19.69, abs=0.01)


# --- the x2 guard's energy ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    'h1, h2, outcome',
    [
        (6879852.0, 6879123.0, GuardOutcome.SAME),  # 30963: |dh| 729 < 687912
        (0.95, 1.06, GuardOutcome.CHANGED),  # crosses the separatrix
        (0.2, 0.3, GuardOutcome.SAME),  # |dh| 0.1, absolute below h = 1 (unchanged rule)
        (0.2, 0.35, GuardOutcome.CHANGED),  # |dh| 0.15 > 0.1
        (20.0, 21.5, GuardOutcome.SAME),  # 1.5 < 0.1 x 20
        (20.0, 22.5, GuardOutcome.CHANGED),  # 2.5 > 2.0
        (1.05, 1.2, GuardOutcome.CHANGED),  # 0.15 > 0.1 x 1.05
    ],
)
def test_energy_guard_relative_far_above_the_separatrix(h1, h2, outcome):
    assert _compare_energy(quantities(h_med=h1), quantities(h_med=h2, cutoff=40.0), S) == outcome


# --- the benchmark rows (author's Q25: certain) ---------------------------------------------------------------------

AUTHOR_CERTAIN = {
    # example_id: far ratio by hand (inf: no zero of the rate at any a > 0)
    'mmr2__16397__29M+50+0-79': math.inf,
    'mmr2__29744__23M-49+0+26': 48.7,
    'mmr2__30963__14M-25+0+11': 58.9,
    'mmr2__35512__11J+4+0-15': math.inf,
    'mmr2__35751__23M+41+0-64': math.inf,
    'mmr2__44662__54M-89+0+35': 93.4,
    'mmr2__49203__76J-23+0-53': 37.4,
    'mmr3__43982__4M+1S-9+0+0+4': 10.8,
}
STAY_HIGH = (
    'mmr3__88710_c4__3E+1M-4+0+0+0',  # far 0.5: a jumps across a*, the angle reverses (46 % of raw steps back)
    'mmr2__88710_c3__49M-31+0-18',  # alias, clearance 25.0 < 33.1 deg
    'mmr2__88710_c7__32E-29+0-3',  # alias, clearance 12.7 < 45.3 deg
)


def _verdict(example_id):
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    if not io.example_path(example_id).exists():
        pytest.skip('benchmark data not available')
    return analyse_mmr(io.load_series(example_id), S)


@pytest.mark.parametrize('example_id', sorted(AUTHOR_CERTAIN))
def test_author_rows_are_certain(example_id):
    analysis = _verdict(example_id)
    v = analysis.verdict
    assert (str(v.status), str(v.angle_status), str(v.confidence)) == ('non_resonant', 'circulation', 'certain')
    expected = AUTHOR_CERTAIN[example_id]
    assert analysis.sampling.far_ratio == (expected if math.isinf(expected) else pytest.approx(expected, rel=0.02))
    assert v.margins['L8'].margin in (Margin.NOT_APPLICABLE, Margin.FULL)


@pytest.mark.parametrize('example_id', STAY_HIGH)
def test_near_or_narrow_branch_rows_stay_high(example_id):
    v = _verdict(example_id).verdict
    assert (str(v.status), str(v.confidence)) == ('non_resonant', 'high')
