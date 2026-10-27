"""The wiring of `analyse_mmr` (W3): subresonances, owner, zone and the filter guard in their fixed order, the derived
row of an accepted subresonance, the caps of SPEC §6.4, the split of a run at its regular cycles, and the planets of
an example completed from its run.

Synthetic tests carry hand-computed numbers; the real-record ones use the benchmark data (git-ignored, skipped when
absent) and compare with an independent source where one exists: the integration itself saved the member angle
9M-16+2+5 of 31935, and the owners come from the methodichka and the independent Codex check (owner tests).
"""

from functools import lru_cache

import numpy as np
import pytest

from resonances.resonance.analysis.confidence import Cap
from resonances.resonance.analysis.geometry import RejectReason, TurningPoints, find_episodes, regular_part
from resonances.resonance.analysis.mmr import decide, io
from resonances.resonance.analysis.mmr.decide import StatusDecision
from resonances.resonance.analysis.mmr.models import ResonanceSpec, SubresonanceCandidate, SubresonanceResult, SubresonanceTrigger
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import CheckValue, Flag, LibrationOrigin, Reason, Status

S = settings_from_config()
NAN = float('nan')


# --- regular part of a run ----------------------------------------------------------------------------------------


def test_regular_part_by_hand():
    # full periods t[i+2] - t[i]: 10 6 2 2 2 2 2 15, median 2, ratio 2 -> cycles 2..6 regular -> TPs 2..8
    assert regular_part(np.array([0, 5, 10, 11, 12, 13, 14, 15, 16, 30.0]), 2.0) == (2, 8)
    assert regular_part(np.arange(10.0), 2.0) == (0, 9)  # uniform: the whole run
    assert regular_part(np.array([0.0, 1.0]), 2.0) is None


def _modulated_run():
    """4 slow cycles (10 kyr) while the centre slides from 6 to 0 rad, then 20 cycles of 1 kyr about 0; amplitude 1."""
    times = np.concatenate([np.arange(0, 35001, 5000.0), 35000 + 500 * np.arange(1, 41)])
    centre = np.where(times <= 35000, 6 - 6 * times / 35000, 0.0)
    kinds = np.where(np.arange(len(times)) % 2 == 0, 1, -1)
    values = centre + kinds * 1.0
    tp = TurningPoints(np.arange(len(times)), times, values, kinds, 0.01)
    return np.linspace(0, 60000, 60001), tp


def test_a_moving_run_is_judged_on_its_regular_cycles():
    t, tp = _modulated_run()
    episodes = find_episodes(t, tp, (0.0, 60000.0), S)
    # whole run: first full-cycle centre c(5 kyr) = 5.14 +- 1 against 0 +- 1 does not overlap -> centre_moves; the
    # regular cycles start at the TP at 35 kyr (t[9] - t[7] = 1 kyr; t[8] - t[6] = 5.5 kyr > 2 x the 1-kyr median) and
    # are stationary; the slow prefix alone (c 5.14 -> 1.71, shift 3.43 > 1 + 1) stays rejected
    accepted = [e for e in episodes if e.accepted]
    assert len(accepted) == 1 and accepted[0].tp_times_yr[0] == 35000.0 and accepted[0].tp_times_yr[-1] == 55000.0
    assert str(accepted[0].shape) == 'stationary'
    assert episodes[0].reject_reason == RejectReason.CENTRE_MOVES and episodes[0].tp_times_yr[-1] == 30000.0
    assert [e.episode_id for e in episodes] == list(range(len(episodes)))


def test_a_drift_with_regular_cycles_stays_rejected():
    """Uniform cycles whose centre slides all the way: nothing to split, still one centre_moves run."""
    times = 500.0 * np.arange(60)
    kinds = np.where(np.arange(60) % 2 == 0, 1, -1)
    tp = TurningPoints(np.arange(60), times, -times / 3000 + kinds * 1.0, kinds, 0.01)
    episodes = find_episodes(np.linspace(0, 30000, 3001), tp, (0.0, 30000.0), S)
    assert len(episodes) == 1 and episodes[0].reject_reason == RejectReason.CENTRE_MOVES


# --- caps of the subresonance check (SPEC §6.4 step 3, §6.3) -------------------------------------------------------


def _scan(triggers=('T1',), tie=False, theory=True, check=CheckValue.FAIL, flags=(), accepted=True, undetermined=False):
    member = SubresonanceCandidate(
        ResonanceSpec.from_key('9M-16+2+5'), 2, False, 0.1, 0.0, 0.1, 0.0, 0.94, 1.0, NAN, NAN, NAN, 1, accepted, ''
    )
    return SubresonanceResult(
        triggers=tuple(SubresonanceTrigger(t) for t in triggers),
        support=(),
        candidates=[member],
        accepted=member if accepted else None,
        check=check,
        composite=False,
        undetermined=undetermined,
        tie_by_order=tie,
        theory_measured=theory,
        sigma0_phi0_coherence=0.59,
        composite_rms_rad=NAN,
        single_rms_rad=NAN,
        binomial_mode=0,
        drift_pred_rad_yr=NAN,
        drift_meas_rad_yr=NAN,
        plane_offset_rad=NAN,
        flags=list(flags),
    )


SUB = StatusDecision(Status.NON_RESONANT, Reason.SUBRESONANCE_LIBRATES, LibrationOrigin.SUBRESONANCE, '', '9M-16+2+5')
KIN = StatusDecision(Status.NON_RESONANT, Reason.KINEMATIC_LIBRATION, LibrationOrigin.KINEMATIC, 'varpi_term', '')


def test_subresonance_caps():
    assert decide.subresonance_caps(SUB, _scan()) == []  # A-E with D confirmed: no cap
    assert decide.subresonance_caps(SUB, _scan(triggers=('T6',))) == [str(Cap.SUBRESONANCE_T6_ONLY)]
    assert decide.subresonance_caps(SUB, _scan(triggers=('T1', 'T6'))) == []
    assert decide.subresonance_caps(SUB, _scan(tie=True)) == [str(Cap.SUBRESONANCE_TIE)]
    assert decide.subresonance_caps(SUB, _scan(theory=False)) == [str(Cap.SUBRESONANCE_THEORY_UNCONFIRMED)]
    # a scan that could not decide caps any verdict (the kinematic one here): the mechanism is not excluded
    missing = _scan(check=CheckValue.NOT_MEASURED, flags=[str(Flag.PLANETS_MISSING)], accepted=False)
    assert decide.subresonance_caps(KIN, missing) == [str(Cap.PLANETS_MISSING)]
    beta = _scan(check=CheckValue.NOT_MEASURED, accepted=False, undetermined=True)
    assert decide.subresonance_caps(KIN, beta) == [str(Cap.SUBRESONANCE_UNDETERMINED)]
    assert decide.subresonance_caps(KIN, _scan(check=CheckValue.PASS, accepted=False)) == []
    assert decide.subresonance_caps(KIN, None) == []


def test_no_provisional_flag_is_left():
    assert 'provisional' not in {str(flag) for flag in Flag}


# --- real records ---------------------------------------------------------------------------------------------------


def _need(example):
    if not (io.example_path(example) / 'series.csv.gz').exists():
        pytest.skip('benchmark data not available')


@lru_cache(maxsize=None)
def _analysis(example):
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    _need(example)
    return analyse_mmr(io.load_series(example), S)


def _wrapped_difference(x, y):
    return np.abs((np.asarray(x) - np.asarray(y) + np.pi) % (2 * np.pi) - np.pi)


def test_31935_subresonance_gets_a_derived_row_of_its_own():
    analysis = _analysis('mmr2__31935__9M-16+0+7')
    verdict = analysis.verdict
    # rule C (W7 'ug22'): the catalogue row takes the status of its member
    assert verdict.status == Status.TRANSIENT and verdict.libration_origin == LibrationOrigin.SUBRESONANCE
    assert verdict.owner == '9M-16+2+5' and analysis.subresonances.accepted.spec.key == '9M-16+2+5'
    (derived,) = analysis.derived
    # its angle is the one the integration saved for the member (an independent source)
    saved = io.load_series('mmr2__31935__9M-16+2+5') if io.example_path('mmr2__31935__9M-16+2+5').exists() else None
    if saved is not None:
        assert np.max(_wrapped_difference(derived.derived_series.angle_wrapped, saved.angle_wrapped)) < 1e-4
    row = derived.summary_row()
    assert row['added_by'] == 'subresonance_check' and row['parent'] == '9M-16+0+7' and row['resonance'] == '9M-16+2+5'
    # the member librates 25-83 kyr (SISTERS): a transient, judged without scans of its own
    assert derived.verdict.status == Status.TRANSIENT
    assert derived.verdict.checks['check_mmr_subresonances'] == CheckValue.NOT_APPLICABLE
    assert derived.verdict.checks['check_mmr_owner'] == CheckValue.NOT_APPLICABLE
    assert analysis.multiplet_status == derived.multiplet_status == Status.TRANSIENT
    from resonances.resonance.analysis.mmr.models import SUMMARY_FIXED_COLUMNS

    assert set(SUMMARY_FIXED_COLUMNS) | {'status', 'angle_status', 'multiplet_status', 'settings_hash'} <= set(row)
    for entry in derived.verdict.decision_path:  # every number the decision read is in the summary row
        assert not entry.column or entry.column in row, entry.column


def test_82831_circulates_and_its_subresonance_is_named():
    """Order 11 against MATRIX_2BODY_ORDER_MAX 10: the catalogue's cap must not remove the members of order q."""
    analysis = _analysis('mmr2__82831__13M-24+0+11')
    verdict = analysis.verdict
    assert str(verdict.final_regime) == 'circulation' and verdict.status == Status.TRANSIENT  # rule C: the member's
    assert verdict.libration_origin == LibrationOrigin.SUBRESONANCE and verdict.owner == '13M-24+2+9'


@pytest.mark.parametrize(
    'example, owner',
    [
        ('mmr3__1997_GK3__3M+5S-3+0+0-5__rerun', '10E-17+0+7'),  # needs Earth from the run's planets file
        ('mmr3__117306__1V+5J-7+0+0+1', '11M-20+0+9'),
    ],
)
def test_owner_names_the_origin(example, owner):
    analysis = _analysis(example)
    assert analysis.verdict.libration_origin == LibrationOrigin.INDUCED and analysis.verdict.owner == owner
    assert analysis.verdict.status == Status.NON_RESONANT and analysis.owner.check == CheckValue.FAIL


def test_owner_outside_the_window_does_not_name_the_origin():
    """A circulation: 7E-23 librates only 0.7-6.4 kyr of the 20-kyr record (cover < MMR_OWNER_COVER_MIN, F7 of W20):
    its key stays in `owner`, the origin is not induced."""
    analysis = _analysis('mmr3__2000_YG4__2R-8E-1+0+0+7__rerun')
    verdict = analysis.verdict
    assert verdict.status == Status.NON_RESONANT and verdict.owner == '7E-23+0+16'
    assert verdict.libration_origin != LibrationOrigin.INDUCED and 'owner_outside_episode' in verdict.flags
    assert analysis.owner.cover < 0.5


def test_planets_of_an_example_completed_from_its_run():
    example = 'mmr3__1997_GK3__3M+5S-3+0+0-5__rerun'
    _need(example)
    series = io.load_series(example)
    assert {'Earth', 'Mars', 'Saturn'} <= set(series.planets)  # the example's own file holds Mars and Saturn only


def test_filter_guard_columns_and_its_effect():
    analysis = _analysis('mmr2__65366__6M-11+0+5')
    numbers = analysis.numbers_row()
    for column in ('flt_x1_n_tp', 'flt_x2_n_tp', 'flt_x1_h_med', 'flt_x2_h_med', 'flt_sensitive_quantities'):
        assert column in numbers
    assert 'flt_x2_angle_status' not in numbers and 'not_computed' not in analysis.summary_row()
    for column in ('geo_episode_centre_travel_max_rad', 'geo_final_centre_travel_rad', 'geo_final_circulation_P'):
        assert column in numbers
    if numbers['flt_filter_sensitive']:
        assert str(Flag.FILTER_SENSITIVE) in analysis.verdict.flags
        assert analysis.verdict.checks['check_mmr_filter'] == CheckValue.NOT_MEASURED


# --- alias far from any commensurability (requests-w2-zone-rules.md §5) ---------------------------------------------


def test_no_commensurability_by_hand():
    from types import SimpleNamespace

    from resonances.resonance.analysis.mmr.sampling import no_commensurability

    # 29M+50+0-79: nu = 29 n_M + 50 n > 0 for every a; n* = 1.5 - (29 x 3.34 + 50 x 1.5) / 50 = -1.937 <= 0
    assert no_commensurability(SimpleNamespace(body_coefficient=50, rate_expected_rad_yr=29 * 3.34 + 50 * 1.5, mean_motion_rad_yr=1.5))
    # 3J-2: nu(<a>) = 0.001 rad/yr, k = -2: n* = 0.6 + 0.0005 > 0, a commensurability exists
    assert not no_commensurability(SimpleNamespace(body_coefficient=-2, rate_expected_rad_yr=0.001, mean_motion_rad_yr=0.6))
    assert not no_commensurability(SimpleNamespace(body_coefficient=0, rate_expected_rad_yr=1.0, mean_motion_rad_yr=0.6))


def test_an_alias_without_any_commensurability_is_non_resonant():
    analysis = _analysis('mmr2__16397__29M+50+0-79')  # author: non_resonant / circulation
    assert analysis.verdict.status == Status.NON_RESONANT and analysis.verdict.reason == Reason.ALIAS
    assert analysis.sampling.far_ratio == float('inf')
