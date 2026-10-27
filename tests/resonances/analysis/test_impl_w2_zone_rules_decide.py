"""W2 zone-rules: the decision rules of DELTA 13 (decide.py, confidence.py).

- 13b: the status follows the observed behaviour; the zone, the energies and episodes_in_separatrix_zone are flags and columns.
- 13c: an angle whose centre runs a full turn is not librating there; the last regime is a circulation (82831).
- 13a: a conclusion that changes at the cutoff x2 makes the checks that read it not_measured (filter_sensitive).
- the alias rule (DECISIONS 2026-10-15) and the physics margins of the cap all_physics_not_measured.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from resonances.resonance.analysis import confidence as conf
from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.ledger import Ledger, Margin
from resonances.resonance.analysis.mmr import decide
from resonances.resonance.analysis.mmr.decide import AngleDecision, StatusDecision
from resonances.resonance.analysis.mmr.models import StepVerdict
from resonances.resonance.analysis.mmr.preprocess import DecisiveQuantities, compare_quantities
from resonances.resonance.analysis.pendulum import (
    EPISODES_IN_SEPARATRIX_ZONE,
    CycleClass,
    CycleEnergy,
    EnergyAnalysis,
    SeparatrixMechanism,
    Zone,
    ZoneMethod,
)
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import (
    AngleStatus,
    AngleSubtype,
    CheckValue,
    Confidence,
    Flag,
    LibrationOrigin,
    MarginClass,
    MarginValue,
    Reason,
    Status,
)
from tests.resonances.analysis.builders import make, make_centre, make_energy, make_episode, make_geometry, make_response
from tests.resonances.analysis.builders import make_sampling
from tests.resonances.analysis.builders import make_turning_points
from tests.resonances.analysis.test_impl_preprocess_geometry_geometry import ONE_DEG, filtered

S = settings_from_config()
SAMPLING = make_sampling(
    unwrap_resid_p99_rad=0.0, unwrap_resid_max_rad=0.0, unwrap_ok=True, undefined_phase=False, verdict='resolved', alias_numpy_differs=False
)


def _ledger():
    return Ledger.from_settings(S)


# --- 13c: centre turns and the last regime ---------------------------------------------------------


def _geometry_with_centre(passes, t, centre, episodes):
    return make_geometry(centre=make_centre(t_yr=t, centre_rad=centre, passes=passes), runs=episodes)


def _episode(period, start, end):
    return make_episode(tp_times_yr=np.array([start, end]), period_med_yr=period)


def test_centre_travel_inside_an_interval_by_its_monotone_passes():
    t = np.linspace(0, 100, 101)
    centre = np.where(t < 50, 0.04 * t, 2.0 - 0.1 * (t - 50))  # up by 2 rad, then down by 5 rad
    passes = [g.MonotonePass(0.0, 50.0, 2.0), g.MonotonePass(50.0, 100.0, -5.0)]
    geometry = _geometry_with_centre(passes, t, centre, [])
    assert decide.centre_travel_in(geometry, 0, 100) == pytest.approx(5.0)
    assert decide.centre_travel_in(geometry, 0, 50) == pytest.approx(2.0)
    assert decide.centre_travel_in(geometry, 40, 70) == pytest.approx(2.0)  # 0.4 up, then 2.0 down
    assert decide.centre_travel_in(geometry, 200, 300) == 0.0


def test_an_episode_whose_centre_turns_inside_is_not_a_libration():
    t = np.linspace(0, 100000, 1001)
    centre = np.where(t < 30000, 0.0, 7.3 * (t - 30000) / 41000)  # 82831: +7.3 rad from 30 to 71 kyr
    passes = [g.MonotonePass(0.0, 30000.0, 0.0), g.MonotonePass(30000.0, 100000.0, 7.3 * 70000 / 41000)]
    turning = _episode(1054.0, 12000.0, 71000.0)
    early = _episode(6773.0, 1900.0, 8700.0)
    geometry = _geometry_with_centre(passes, t, centre, [early, turning])
    assert decide.librating_episodes(geometry, S) == [early]
    fast = _episode(S.mmr_libration_period_min_yr / 2, 0.0, 1000.0)
    assert decide.librating_episodes(_geometry_with_centre(passes, t, centre, [fast]), S) == []


def test_the_last_regime_is_a_circulation_when_the_centre_turns_after_the_episodes():
    t = np.arange(0, 20000, 1.0)
    libration = np.pi + np.radians(60) * np.sin(2 * np.pi * t / 500)
    circulation = np.pi + 2 * np.pi * 5 * (t - 10000) / 10000  # 5 turns in the last 10 kyr = 20 periods
    geometry = g.measure_geometry(filtered(t, np.where(t < 10000, libration, circulation), 60.0), ONE_DEG, S)
    angle = decide.decide_angle(SAMPLING, False, geometry, None, S, _ledger())
    assert (angle.angle_status, angle.rule) == (AngleStatus.CIRCULATION, 'G1c') and angle.episode_id is not None
    # the status reads the episodes: a transient by the observed behaviour
    assert decide.status_basis(angle).angle_status == AngleStatus.EPISODIC
    # a visible exit inside the last period stays the episode's (libration, exit_suspected)
    exit_only = np.pi + 2 * np.pi * (t - 19700) / 300
    late = g.measure_geometry(filtered(t, np.where(t < 19700, libration, exit_only), 60.0), ONE_DEG, S)
    assert decide.decide_angle(SAMPLING, False, late, None, S, _ledger()).angle_status == AngleStatus.LIBRATION


def _response():
    return make_response(
        a_response=CheckValue.PASS,
        a_amplitude=CheckValue.PASS,
        slow_libration=CheckValue.NOT_APPLICABLE,
        b1_ratio=0.5,
        slow_adot_r2=0.1,
        force_r2_h1=0.8,
        force_r2_h2=0.8,
        centre_rate_rad_yr=0.0,
        centre_rate_err_rad_yr=1.0,
        force_law='single',
        multiplet=None,
        phi0_max_dev_rad=0.1,
        phi0_tol_rad=0.5,
        a_response_thin=False,
    )


def _energy(chaos=False, classes=(), with_slow=CheckValue.PASS, mechanism=SeparatrixMechanism.NONE):
    return EnergyAnalysis(
        None, None, 1.0, 1.0, cycle_classes=list(classes), with_slow=with_slow, episodes_in_separatrix_zone=chaos, mechanism=mechanism
    )


def test_a_last_regime_circulation_keeps_the_status_of_its_episodes():
    geometry = make_geometry(turning_points=make_turning_points([1, 2, 3]))
    angle = AngleDecision(AngleStatus.CIRCULATION, AngleSubtype.NONE, 'G1c', 0)
    decision = decide.decide_status(angle, SAMPLING, geometry, _energy(), _response(), None, None, None, None, S, _ledger())
    assert (decision.status, decision.reason, decision.origin) == (Status.TRANSIENT, Reason.DYNAMIC_EPISODE, LibrationOrigin.DYNAMIC)
    plain = AngleDecision(AngleStatus.CIRCULATION, AngleSubtype.NONE, 'G4')
    other = decide.decide_status(plain, SAMPLING, geometry, _energy(), _response(), None, None, None, None, S, _ledger())
    assert (other.status, other.reason) == (Status.NON_RESONANT, Reason.CIRCULATION)


# --- 13b: the separatrix physics are flags -----------------------------------------------------------


@pytest.mark.parametrize('status, expected', [(AngleStatus.LIBRATION, Status.RESONANT), (AngleStatus.EPISODIC, Status.TRANSIENT)])
def test_episodes_in_separatrix_zone_is_a_flag_not_a_status(status, expected):
    geometry = make_geometry(turning_points=make_turning_points([1, 2, 3]))
    angle = AngleDecision(status, AngleSubtype.NONE, 'G2', 0)
    for chaos in (False, True):
        energy = _energy(chaos=chaos, classes=[CycleClass.ZONE] * 3)
        decision = decide.decide_status(angle, SAMPLING, geometry, energy, _response(), None, None, None, None, S, _ledger())
        assert decision.status == expected
        assert (EPISODES_IN_SEPARATRIX_ZONE in decision.flags) == chaos


def test_episodes_in_separatrix_zone_caps_a_transient_at_medium_and_replaces_the_onset_test():
    assert conf.CAP_CEILING[conf.Cap.EPISODES_IN_SEPARATRIX_ZONE] == Confidence.MEDIUM
    angle = AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, 'G2', 0)
    transient = StatusDecision(Status.TRANSIENT, Reason.DYNAMIC_EPISODE, LibrationOrigin.DYNAMIC, '', '')
    geometry = make_geometry()
    chaos = _energy(chaos=True, mechanism=SeparatrixMechanism.UNEXPLAINED)
    caps = decide.collect_caps(decide.tree_view(angle, None, chaos), transient, geometry, {}, S)
    assert str(conf.Cap.EPISODES_IN_SEPARATRIX_ZONE) in caps and str(conf.Cap.TRANSIENT_UNEXPLAINED) not in caps
    resonant = StatusDecision(Status.RESONANT, Reason.DYNAMIC_LIBRATION, LibrationOrigin.DYNAMIC, '', '')
    libration = AngleDecision(AngleStatus.LIBRATION, AngleSubtype.STATIONARY, 'G2', 0)
    assert str(conf.Cap.EPISODES_IN_SEPARATRIX_ZONE) not in decide.collect_caps(
        decide.tree_view(libration, None, chaos), resonant, geometry, {}, S
    )
    # the onset test assumes a capture from a regime without the a-wave: not inside one chaotic layer
    physics = SimpleNamespace(a_band_std_before_au=1.0, a_band_std_in_au=1.0)
    assert decide._o1(physics, chaos, _ledger(), S).margin == Margin.NOT_APPLICABLE
    assert decide._o1(physics, _energy(), _ledger(), S).margin == Margin.FAIL


def _circulating_energy(widths):
    """A circulation whose turns run `widths` zone widths above the zone (h_lo 0.9, h_hi 1.1: width 0.2)."""
    zone = make(Zone, method=ZoneMethod.NUMERIC, h_lo=0.9, h_hi=1.1)
    turn = make(CycleEnergy, rotation=True, h=1.1 + widths * 0.2)
    return make_energy(with_slow=CheckValue.PASS, zone=zone, cycles=[turn])


def test_near_separatrix_energy_flag_of_a_circulation():
    near = _circulating_energy(0.8)
    far = _circulating_energy(S.separatrix_near_zone_widths * 3)
    assert near.turn_zone_widths == pytest.approx(0.8)
    assert decide.near_separatrix_energy(near, S, _ledger())
    assert not decide.near_separatrix_energy(far, S, _ledger())
    assert not decide.near_separatrix_energy(None, S, _ledger())
    geometry = make_geometry()
    angle = AngleDecision(AngleStatus.CIRCULATION, AngleSubtype.NONE, 'G4')
    energy = near
    decision = decide.decide_status(angle, SAMPLING, geometry, energy, None, None, None, None, None, S, _ledger())
    assert decision.status == Status.NON_RESONANT and str(Flag.NEAR_SEPARATRIX_ENERGY) in decision.flags


# --- the alias rule ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    'verdict, far_ratio, status',
    [
        (StepVerdict.ALIASED_UNAMBIGUOUS, 50.0, Status.NON_RESONANT),  # one branch, far from a*: a circulation
        (StepVerdict.ALIASED_UNAMBIGUOUS, 1.0, Status.INVALID),  # one branch but near a*: a libration could hide
        (StepVerdict.AMBIGUOUS, 50.0, Status.INVALID),  # several branches: the rate is not known
        (StepVerdict.ALIASED_UNAMBIGUOUS, float('nan'), Status.INVALID),  # a* unknown: not shown to be far
    ],
)
def test_alias_is_a_circulation_only_with_one_branch_far_from_the_resonance(verdict, far_ratio, status):
    sampling = make_sampling(verdict=verdict, far_ratio=far_ratio)
    angle = AngleDecision(AngleStatus.CIRCULATION, AngleSubtype.ALIAS, 'G4')
    decision = decide.decide_status(angle, sampling, make_geometry(), None, None, None, None, None, None, S, _ledger())
    assert (decision.status, decision.reason) == (status, Reason.ALIAS)


# --- 13a: the x2 guard ---------------------------------------------------------------------------------


def _quantities(n_tp=12, n_episodes=1, stall=False, h_med=0.5, cutoff=20.0):
    return DecisiveQuantities(cutoff, n_tp, n_episodes, stall, 0.0, h_med)


def test_the_guard_compares_numbers_within_their_tolerances():
    same = compare_quantities(_quantities(), _quantities(n_tp=10, h_med=0.6, cutoff=40.0), S)
    assert not same.sensitive  # 2 TPs fewer (MMR_FILTER_GUARD_TP_ABS) and dh = 0.1 (MMR_FILTER_GUARD_H_TOL): unchanged
    lost = compare_quantities(_quantities(), _quantities(n_tp=2, n_episodes=0, cutoff=40.0), S)
    assert lost.changed == ('turning_points', 'episodes')
    energy = compare_quantities(_quantities(h_med=0.9), _quantities(h_med=1.1, cutoff=40.0), S)
    assert energy.changed == ('energy',)
    unknown = compare_quantities(_quantities(h_med=float('nan')), _quantities(h_med=1.1, cutoff=40.0), S)
    assert unknown.changed == ('energy',)  # measured at one cutoff only: a change (W3-FIXES F1)
    assert compare_quantities(_quantities(), None, S).sensitive is None  # nothing compared: not checked, not False
    columns = lost.to_columns()
    assert columns['flt_x1_n_tp'] == 12 and columns['flt_x2_n_tp'] == 2 and columns['flt_x2_cutoff_yr'] == 40.0
    assert columns['flt_filter_sensitive'] and columns['flt_sensitive_quantities'] == 'turning_points;episodes'


def test_a_sensitive_record_loses_the_checks_that_read_the_changed_quantity():
    checks = {name: CheckValue.PASS for name in ('check_mmr_filter', 'check_mmr_one_pendulum', 'check_mmr_potential')}
    checks['check_mmr_onset'] = CheckValue.NOT_APPLICABLE
    margins = {m: MarginValue(Margin.FULL, 1.0, MarginClass.DECISIVE) for m in ('L5', 'R0', 'L1')}
    flags = []
    decide.apply_filter_sensitivity(checks, margins, compare_quantities(_quantities(), _quantities(n_tp=2, n_episodes=0), S), flags)
    assert flags == [str(Flag.FILTER_SENSITIVE)]
    assert checks['check_mmr_filter'] == checks['check_mmr_one_pendulum'] == CheckValue.NOT_MEASURED
    assert checks['check_mmr_onset'] == CheckValue.NOT_APPLICABLE and checks['check_mmr_potential'] == CheckValue.PASS
    assert margins['L5'].margin == Margin.NOT_MEASURED and margins['L5'].value == 1.0
    assert margins['R0'].margin == margins['L1'].margin == Margin.FULL
    untouched = dict(checks)
    decide.apply_filter_sensitivity(untouched, margins, compare_quantities(_quantities(), _quantities(), S), [])
    assert untouched == checks


# --- the physics margins of the cap all_physics_not_measured -------------------------------------------


def test_the_owner_search_and_the_strength_are_physics_identity_and_b1_are_not():
    cells = conf.ConfidenceCells()
    nm, full, thin = (MarginValue(Margin(x), None, MarginClass.DECISIVE) for x in ('not_measured', 'full', 'thin'))
    small_amplitude = {'L3': full, 'L5': thin, 'L4': nm, 'R0': nm}  # identity, one pendulum, force law and R0 not measured
    capped = conf.assess(Status.RESONANT, 'libration', 'dynamic_libration', small_amplitude, [], [], decide.PHYSICS_MARGINS, cells)
    assert str(conf.Cap.ALL_PHYSICS_NOT_MEASURED) in capped.caps and capped.confidence == Confidence.MEDIUM
    searched = dict(small_amplitude, W=full)  # the owner search ran and found no owner
    assessed = conf.assess(Status.RESONANT, 'libration', 'dynamic_libration', searched, [], [], decide.PHYSICS_MARGINS, cells)
    assert str(conf.Cap.ALL_PHYSICS_NOT_MEASURED) not in assessed.caps and assessed.confidence == Confidence.HIGH
    strength = dict(small_amplitude, F=thin)
    assert (
        conf.assess(Status.RESONANT, 'libration', 'dynamic_libration', strength, [], [], decide.PHYSICS_MARGINS, cells).confidence
        == Confidence.HIGH
    )
    assert 'L3' not in decide.PHYSICS_MARGINS and 'L5' not in decide.PHYSICS_MARGINS
