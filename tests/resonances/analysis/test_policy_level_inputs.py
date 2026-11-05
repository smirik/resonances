"""The inputs of the minimal policy's level (`policy.level_inputs`, column `pol_level_inputs`), the step that decided
the status (`pol_decided_by`), the checks behind an objection (`pol_objection_check`) and the subtype of a word the
policy changes. None of them may change a status or a level: the level is the lowest ceiling of its inputs.

Every expected text and level is read off the table of docs/physics/confidence.md (minimal policy) by hand for the
defaults CONFIDENCE_MEDIUM_CYCLES = 1.5, CONFIDENCE_HIGH_CYCLES = 2, CONFIDENCE_CERTAIN_CYCLES = 3,
CONFIDENCE_CERTAIN_CORE_CYCLES = 3 and CONFIDENCE_GEOMETRY_ONLY_MAX = low; the grids compare the inputs with
`policy.level` and `policy._final_level`, the functions that set the level."""

import itertools
import math

import numpy as np
import pytest

from resonances.resonance.analysis import policy
from resonances.resonance.analysis.ledger import Ledger, Op
from resonances.resonance.analysis.mmr import decide, models
from resonances.resonance.analysis.mmr.background_wave import FORCED, BackgroundWave
from resonances.resonance.analysis.mmr.strength import DIRECT_METHODS
from resonances.resonance.analysis.secular.models import GateResult
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import (
    CONFIDENCE_ORDER,
    AngleStatus,
    AngleSubtype,
    CheckValue,
    Confidence,
    Flag,
    LibrationOrigin,
    Reason,
    ResonanceType,
    Status,
    Verdict,
)
from tests.resonances.analysis.builders import (
    make,
    make_centre,
    make_episode,
    make_geometry,
    make_long_period_analysis,
    make_mmr_analysis,
    make_turning_points,
    make_verdict,
)

SETTINGS = AnalysisSettings()
NAN = math.nan
C, H, M, L = Confidence.CERTAIN, Confidence.HIGH, Confidence.MEDIUM, Confidence.LOW


def ev(n=40.0, margin=policy.M_FULL, veto_state=policy.V_CLEAN, stable=True, **fields):
    fields.setdefault('core_cycles', n)
    return policy.Evidence(n_cycles=n, margin=margin, veto_state=veto_state, stable=stable, **fields)


# --------------------------------------------------------------------------------------------- the inputs by hand


@pytest.mark.parametrize(
    'status, evidence, text, expected',
    [
        # 4.12 cycles >= 3, the decisive episode's 4 >= 3, but M not measured and S thin: high (M and S)
        (
            Status.RESONANT,
            ev(4.12, policy.M_NOT_MEASURED, stable=False, core_cycles=4.0, force=policy.F_SHOWN),
            'N:4.12>=3:certain;core:4>=3:certain;M:not_measured:high;S:thin:high;V:clean:certain;F:shown:certain',
            H,
        ),
        # one closed cycle (1 < 1.5): low whatever the rest
        (Status.TRANSIENT, ev(1.0), 'N:1<1.5:low;core:1<3:high;M:full:certain;S:full:certain;V:clean:certain', L),
        # N 3.09 with the covered edges but 2 observed cycles in the episode (secular 518): high by core
        (Status.RESONANT, ev(3.09, core_cycles=2.0), 'N:3.09>=3:certain;core:2<3:high;M:full:certain;S:full:certain;V:clean:certain', H),
        # a needed veto not measured: medium; N 2.5 alone would allow high
        (
            Status.RESONANT,
            ev(2.5, veto_state=policy.V_NOT_MEASURED),
            'N:2.5<3:high;core:2.5<3:high;M:full:certain;S:full:certain;V:not_measured:medium',
            M,
        ),
        # an MMR row whose own force is not shown: CONFIDENCE_GEOMETRY_ONLY_MAX (low)
        (
            Status.RESONANT,
            ev(40.0, force=policy.F_NONE),
            'N:40>=3:certain;core:40>=3:certain;M:full:certain;S:full:certain;V:clean:certain;F:none:low',
            L,
        ),
        (Status.NEAR_SEPARATRIX, ev(veto_state=policy.V_AMBIGUOUS), 'status:near_separatrix:medium;V:ambiguous:low', L),
        (Status.NEAR_SEPARATRIX, ev(), 'status:near_separatrix:medium;V:clean:certain', M),
        # V's own ceiling as on a librating row: not_measured holds medium too, partial alone would allow high
        (Status.NEAR_SEPARATRIX, ev(veto_state=policy.V_NOT_MEASURED), 'status:near_separatrix:medium;V:not_measured:medium', M),
        (Status.NEAR_SEPARATRIX, ev(veto_state=policy.V_PARTIAL), 'status:near_separatrix:medium;V:partial:high', M),
        (Status.NON_RESONANT, ev(NAN, veto='owner', veto_thin=True), 'veto:thin(owner):medium;second_capture:none:certain', M),
        (Status.NON_RESONANT, ev(NAN, veto='strength', candidate=True), 'veto:full(strength):high;second_capture:none:certain', H),
        (Status.NON_RESONANT, ev(NAN, candidate=True, circulation_clear=True), 'candidate:drawn:medium;circulation:clear:certain', M),
        (Status.NON_RESONANT, ev(NAN, circulation_clear=True), 'candidate:none:certain;circulation:clear:certain', C),
        (Status.NON_RESONANT, ev(NAN), 'candidate:none:certain;circulation:not_clear:high', H),
        (Status.UNCERTAIN, ev(), '', Confidence.NONE),
        (Status.INVALID, ev(), '', Confidence.NONE),
    ],
)
def test_the_inputs_name_what_the_level_read(status, evidence, text, expected):
    inputs = policy.level_inputs(status, evidence, SETTINGS)
    assert policy.level_inputs_text(inputs) == text
    assert policy.lowest_ceiling(inputs) == expected == policy.level(status, evidence, SETTINGS)


def test_a_fixed_level_and_a_thin_veto_are_inputs():
    # a subresonance member's high: the table alone (V is read only on the table's medium)
    inputs = policy.level_inputs(Status.RESONANT, ev(veto_state=policy.V_AMBIGUOUS), SETTINGS, outcome_level=H)
    assert policy.level_inputs_text(inputs) == 'table:high:high'
    # 5a on near_separatrix: the table's medium, low with V ambiguous
    inputs = policy.level_inputs(Status.NON_RESONANT, ev(veto_state=policy.V_AMBIGUOUS), SETTINGS, outcome_level=M)
    assert policy.level_inputs_text(inputs) == 'table:medium:medium;V:ambiguous:low'
    # the table's medium with V partial or not_measured: V's own ceiling (high, medium), the level stays medium
    for state, text in ((policy.V_PARTIAL, 'V:partial:high'), (policy.V_NOT_MEASURED, 'V:not_measured:medium')):
        inputs = policy.level_inputs(Status.NON_RESONANT, ev(veto_state=state), SETTINGS, outcome_level=M)
        assert policy.level_inputs_text(inputs) == f'table:medium:medium;{text}' and policy.lowest_ceiling(inputs) == M
    # a thin veto on a row that is no vetoed non_resonant caps certain at medium
    inputs = policy.level_inputs(Status.RESONANT, ev(), SETTINGS, veto_thin=True)
    assert inputs[-1].text() == 'thin_veto:thin:medium' and policy.lowest_ceiling(inputs) == M


# --------------------------------------------------------------------------------------------- the contract on grids


def outcome_with(status, level_=None, veto_thin=False):
    return policy._Outcome(status, Reason.CIRCULATION, LibrationOrigin.NONE, level=level_, veto_thin=veto_thin)


def check(status, evidence, level_=None, veto_thin=False):
    inputs = policy.level_inputs(status, evidence, SETTINGS, level_, veto_thin)
    final = policy._final_level(status, outcome_with(status, level_, veto_thin), evidence, SETTINGS)
    assert policy.lowest_ceiling(inputs) == final, (status, evidence, level_, veto_thin, policy.level_inputs_text(inputs))
    if status in (Status.UNCERTAIN, Status.INVALID):
        assert inputs == () and final == Confidence.NONE
    if level_ is None and not veto_thin:
        assert final == policy.level(status, evidence, SETTINGS)
    for entry in inputs:
        assert entry.ceiling in CONFIDENCE_ORDER


N_GRID = (NAN, 0.0, 0.5, 1.0, 1.5, 1.9, 2.0, 2.9, 3.0, 40.0)
V_GRID = policy.V_ORDER


def test_librating_rows_take_the_lowest_ceiling():
    grid = itertools.product(
        (Status.RESONANT, Status.TRANSIENT),
        N_GRID,
        (NAN, 2.0, 3.0),
        (policy.M_FULL, policy.M_THIN, policy.M_NOT_MEASURED),
        V_GRID,
        (True, False),
        ('', policy.F_SHOWN, policy.F_NONE),
        (False, True),
    )
    for status, n, core, margin, veto_state, stable, force, thin in grid:
        check(status, ev(n, margin, veto_state, stable, core_cycles=core, force=force), veto_thin=thin)


def test_the_other_statuses_take_the_lowest_ceiling():
    grid = itertools.product(
        (Status.NEAR_SEPARATRIX, Status.NON_RESONANT, Status.UNCERTAIN, Status.INVALID),
        (NAN, 3.0),
        V_GRID,
        ('', 'owner'),
        (False, True),
        (False, True),
        (False, True),
        (False, True),
        (False, True),
    )
    for status, n, veto_state, veto, ev_thin, second, candidate, clear, thin in grid:
        evidence = ev(
            n, veto_state=veto_state, veto=veto, veto_thin=ev_thin, second_capture=second, candidate=candidate, circulation_clear=clear
        )
        check(status, evidence, veto_thin=thin)


def test_a_fixed_level_takes_the_lowest_ceiling():
    levels = CONFIDENCE_ORDER + (Confidence.NONE,)
    for status, level_, veto_state, thin, n in itertools.product(tuple(Status), levels, V_GRID, (False, True), (NAN, 40.0)):
        if status in (Status.UNCERTAIN, Status.INVALID) and level_ != Confidence.NONE:
            continue  # the table fixes no level there (`_on_member` writes none for an uncertain member)
        check(status, ev(n, veto_state=veto_state), level_, thin)


def test_the_geometry_only_limit_moves_the_f_ceiling():
    settings = AnalysisSettings(confidence_geometry_only_max='medium')
    inputs = policy.level_inputs(Status.RESONANT, ev(force=policy.F_NONE), settings)
    assert inputs[-1].text() == 'F:none:medium'
    assert policy.lowest_ceiling(inputs) == M == policy.level(Status.RESONANT, ev(force=policy.F_NONE), settings)


# --------------------------------------------------------------------------------------------- MMR rows


def episode_geometry(n_tp=7, period=400.0, **fields):
    """One accepted episode of `n_tp` turning points half a period apart in [0, 2000] yr (above the 150-yr floor)."""
    times = 100.0 + np.arange(n_tp) * period / 2
    run = make_episode(
        n_tp,
        episode_id=0,
        tp_times_yr=times,
        period_med_yr=period,
        half_amp_max_rad=0.5,
        coverage_start_gap_p=0.2,
        coverage_end_gap_p=0.2,
    )
    return make_geometry(window=(0.0, 2000.0), runs=[run], period_cycle_yr=period, **fields)


def row(rule, word=AngleStatus.LIBRATION, status=Status.UNCERTAIN, reason=Reason.ORIGIN_UNKNOWN, geometry=None, verdict=None, **fields):
    verdict = make_verdict(status, word, reason=reason, libration_origin=LibrationOrigin.UNKNOWN, **(verdict or {}))
    return make_mmr_analysis(verdict, geometry or episode_geometry(), rule, decisive_episode_id=0, **fields)


def applied(analysis) -> Verdict:
    v = policy.apply_mmr(analysis, SETTINGS).verdict
    assert policy.lowest_ceiling(analysis.policy.level_inputs) == v.confidence  # the inputs explain the level
    assert policy.level_inputs_text(analysis.policy.level_inputs) == v.pol_level_inputs
    assert bool(v.pol_objection_check) == (analysis.policy.veto_state == policy.V_AMBIGUOUS)
    return v


@pytest.mark.parametrize(
    'rule, checks',
    [
        ('5g', 'identity'),
        ('5a', 'force_law;potential'),
        ('5a*', 'force_law'),
        ('5e', 'potential'),
        ('5b', 'potential'),
        ('5f', 'force_line'),
    ],
)
def test_an_objection_names_its_check(rule, checks):
    v = applied(row(rule))
    assert (v.status, v.confidence) == (Status.RESONANT, L)
    assert (v.pol_decided_by, v.pol_objection_check) == (policy.DECIDED_OBJECTION, checks)
    assert 'V:ambiguous:low' in v.pol_level_inputs


def onset_failed_path():
    ledger = Ledger(SETTINGS.confidence_thin_band)
    ledger.compare(decide.O1_ONSET, 0.9, Op.LT, 0.6, '', 'ep_onset_ratio')  # 0.9 >= 0.6: O1 fails
    return list(ledger)


def test_5o_names_the_onset_and_the_strength():
    v = applied(row('5o', word=AngleStatus.EPISODIC, verdict={'decision_path': onset_failed_path()}))
    assert (v.status, v.pol_objection_check) == (Status.TRANSIENT, 'onset')
    strength = make(models.Strength, method=DIRECT_METHODS[0], ratio=40.0, check=CheckValue.FAIL)
    v = applied(row('5o', word=AngleStatus.EPISODIC, strength=strength))
    assert v.pol_objection_check == 'strength'


def test_a_failed_identity_on_a_kept_leaf_is_named():
    # 523899 9N-17, 564160: the tree kept its libration (5c/6), the identity failed: V ambiguous, low, no flag
    v = applied(
        row('5c/6', status=Status.RESONANT, reason=Reason.DYNAMIC_LIBRATION, verdict={'checks': {'check_mmr_identity': CheckValue.FAIL}})
    )
    assert (v.status, v.confidence, v.pol_decided_by, v.pol_objection_check) == (Status.RESONANT, L, policy.DECIDED_TREE, 'identity')
    # at e near 0 the step turns only varpi fast: the failure is expected, no objection (V not measured)
    checks = {'check_mmr_identity': CheckValue.FAIL}
    v = applied(
        row(
            '5c/6',
            status=Status.RESONANT,
            reason=Reason.DYNAMIC_LIBRATION,
            verdict={'checks': checks, 'flags': [str(Flag.VARPI_FAST_LOW_E)]},
        )
    )
    assert v.pol_objection_check == '' and 'V:not_measured:medium' in v.pol_level_inputs


def test_a_kept_libration_is_decided_by_the_tree():
    v = applied(row('5c/6', status=Status.RESONANT, reason=Reason.DYNAMIC_LIBRATION))
    assert (v.status, v.pol_decided_by, v.pol_objection_check) == (Status.RESONANT, policy.DECIDED_TREE, '')


def test_a_geometric_leaf_that_moves_the_status_is_named():
    v = applied(row('5u/6', word=AngleStatus.EPISODIC, status=Status.UNCERTAIN))
    assert (v.status, v.pol_decided_by) == (Status.TRANSIENT, policy.DECIDED_GEOMETRY)


def test_an_upheld_strength_veto_is_the_tree_and_drops_the_old_subtype():
    # 2897 35S-4: the tree's episodic 'recurrent' becomes the angle's moving word; the subtype went with the old word
    strength = make(models.Strength, method=DIRECT_METHODS[0], ratio=1.0e6, three_body=False, eq_legal=True)
    analysis = row(
        '5s',
        word=AngleStatus.EPISODIC,
        status=Status.NON_RESONANT,
        reason=Reason.IMPLAUSIBLE_STRENGTH,
        verdict={'angle_subtype': AngleSubtype.RECURRENT},
        strength=strength,
        extra={'ep_n_tp_cut_half': 2, 'ep_n_tp_cut_double': 7},
    )
    v = applied(analysis)
    assert (v.status, v.veto, v.pol_decided_by) == (Status.NON_RESONANT, policy.VETO_STRENGTH, policy.DECIDED_TREE)
    assert (v.angle_status, v.angle_subtype) == (AngleStatus.DRIFT, AngleSubtype.NONE)
    assert v.v2_verdict.endswith('/episodic')


def test_a_word_the_policy_keeps_keeps_its_subtype():
    v = applied(row('5g', verdict={'angle_subtype': AngleSubtype.STATIONARY}))
    assert (v.angle_status, v.angle_subtype) == (AngleStatus.LIBRATION, AngleSubtype.STATIONARY)


def test_a_three_body_strength_leaf_then_the_forced_wave_says_the_leaf_was_not_upheld():
    # 2012 KL9 5E-2M-10: leaf 5s on a three-body scale (no computed force: not upheld), its one counted episode the
    # forced wave of its background on a centre that passes 1.3 revolutions: veto forced_wave, circulation
    geometry = episode_geometry(centre=make_centre(max_pass_rad=1.3 * 2 * np.pi))
    strength = make(models.Strength, method='eq_scale', ratio=1.0e4, three_body=True, eq_legal=True)
    analysis = row(
        '5s',
        word=AngleStatus.EPISODIC,
        status=Status.NON_RESONANT,
        reason=Reason.IMPLAUSIBLE_STRENGTH,
        geometry=geometry,
        verdict={'angle_subtype': AngleSubtype.RECURRENT},
        strength=strength,
    )
    analysis.background_waves = (BackgroundWave(0, FORCED, '', 1.0, 1.0, 1.0, 20.0, 0.9),)
    v = applied(analysis)
    assert (v.status, v.reason, v.veto) == (Status.NON_RESONANT, Reason.KINEMATIC_LIBRATION, policy.VETO_FORCED_WAVE)
    assert v.pol_decided_by == 'veto:forced_wave;not_upheld:5s'
    assert (v.angle_status, v.angle_subtype) == (AngleStatus.CIRCULATION, AngleSubtype.NONE)
    assert v.pol_level_inputs.startswith('veto:')


def test_a_three_body_strength_leaf_alone_is_a_scale_objection():
    strength = make(models.Strength, method='eq_scale', ratio=1.0e4, three_body=True, eq_legal=True)
    v = applied(row('5s', status=Status.NON_RESONANT, reason=Reason.IMPLAUSIBLE_STRENGTH, strength=strength))
    assert (v.status, v.confidence) == (Status.RESONANT, L)
    assert (v.pol_decided_by, v.pol_objection_check) == ('strength_scale;not_upheld:5s', 'strength_scale')


# --------------------------------------------------------------------------------------------- ZLK rows

PERIOD, SPAN = 5.0e4, 1.0e6


def zlk_row(word, status, reason, outcome, **gate_fields):
    times = 1.0e5 + np.arange(9) * PERIOD / 2
    run = make_episode(
        9, episode_id=0, tp_times_yr=times, period_med_yr=PERIOD, half_amp_max_rad=0.6, coverage_start_gap_p=0.2, coverage_end_gap_p=0.2
    )
    geometry = make_geometry(window=(0.0, SPAN), runs=[run], turning_points=make_turning_points(times), period_cycle_yr=PERIOD)
    verdict = make_verdict(status, word, resonance_type=ResonanceType.LIDOV_KOZAI, reason=reason)
    gate_fields.setdefault('n_cycles', 18.0)
    gate_fields.setdefault('min_cycles', 3.0)
    analysis = make_long_period_analysis(verdict, geometry, GateResult(outcome, **gate_fields), span_yr=SPAN, decisive_episode_id=0)
    analysis.decisive_period_yr = PERIOD
    policy.apply_long_period(analysis, SETTINGS)
    v = analysis.verdict
    assert policy.lowest_ceiling(analysis.policy.level_inputs) == v.confidence
    assert bool(v.pol_objection_check) == (analysis.policy.veto_state == policy.V_AMBIGUOUS)
    return v


def test_a_zlk_veto_the_tree_had_is_upheld():
    v = zlk_row(AngleStatus.LIBRATION, Status.NON_RESONANT, Reason.KINEMATIC_LIBRATION, 'kinematic', rho_max=0.4)
    assert (v.status, v.veto, v.pol_decided_by) == (Status.NON_RESONANT, policy.VETO_ZLK_KINEMATIC, policy.DECIDED_TREE)


def test_a_zlk_kinematic_tree_status_without_a_policy_veto_is_named():
    v = zlk_row(AngleStatus.LIBRATION, Status.NON_RESONANT, Reason.KINEMATIC_LIBRATION, 'undefined_phase', rho_max=1.5)
    assert (v.status, v.veto) == (Status.RESONANT, '')
    assert v.pol_decided_by == 'gate_unavailable;not_upheld:zlk_gate'


def test_a_zlk_confirmed_gate_and_an_objection_are_named():
    v = zlk_row(AngleStatus.LIBRATION, Status.RESONANT, Reason.GEOMETRIC_LIBRATION, 'confirmed')
    assert (v.reason, v.pol_decided_by) == (Reason.DYNAMIC_LIBRATION, policy.DECIDED_GATE_CONFIRMED)
    v = zlk_row(AngleStatus.LIBRATION, Status.RESONANT, Reason.DYNAMIC_LIBRATION, 'undefined_phase', rho_max=0.5)
    assert (v.status, v.confidence, v.pol_decided_by, v.pol_objection_check) == (Status.RESONANT, L, 'objection', 'zlk_free_e')
    v = zlk_row(
        AngleStatus.EPISODIC,
        Status.TRANSIENT,
        Reason.DYNAMIC_EPISODE,
        'kinematic',
        rho_max=1.4,
        rho_episodes=0.8,
        free_turns_episodes=0.1,
        free_turns_min=0.5,
    )
    assert (v.status, v.pol_objection_check) == (Status.TRANSIENT, 'zlk_gate_kinematic')


def test_the_columns_round_trip():
    v = applied(row('5a'))
    back = Verdict.from_flat_dict(v.to_flat_dict())
    assert (back.pol_level_inputs, back.pol_decided_by, back.pol_objection_check) == (
        v.pol_level_inputs,
        v.pol_decided_by,
        v.pol_objection_check,
    )
