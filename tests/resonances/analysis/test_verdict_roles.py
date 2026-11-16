"""A3a (W28; cards g04-6, g06-1, g06-2, g06-3, g05-3): the roles of the other resonance in an MMR verdict.

The `owner` column keeps only an owner or driver that makes the deciding episode; an owner F7/P2 refused goes to
`neighbours` ('KEY t0–t1 before|during|after'), the librating member of the multiplet to `resonant_argument` (rule C),
the term behind the verdict to `cause_role`/`cause_term` ('-4varpi' for a kinematic libration), and the key a statistic
counts once to `commensurability`. No status, angle word or level changes.

The expected strings are written out by hand from the constructions (the interval ends, the coefficients of the keys),
not produced by the code under test; the benchmark rows quote the author's answers of 2026-10-31.
"""

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import analyse, decide, io, owner, roles
from resonances.resonance.analysis.mmr import subresonances as sub
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.decide import AngleDecision
from resonances.resonance.analysis.mmr.models import (
    EpisodePhysics,
    OwnerCandidate,
    OwnerResult,
    ResonanceSpec,
    SubresonanceCandidate,
)
from resonances.resonance.analysis.pendulum import EnergyAnalysis
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import (
    AngleStatus,
    AngleSubtype,
    CauseRole,
    CheckValue,
    Confidence,
    Flag,
    LibrationOrigin,
    OriginDetail,
    Reason,
    ResonanceType,
    Status,
    Verdict,
)
from tests.resonances.analysis.builders import make, make_episode, make_geometry, make_response, make_sampling

S = settings_from_config()
NAN = float('nan')
EPISODIC = AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, 'G2', 1)
SAMPLING = make_sampling(
    unwrap_resid_p99_rad=0.0, unwrap_resid_max_rad=0.0, unwrap_ok=True, undefined_phase=False, verdict='resolved', alias_numpy_differs=False
)
ROLE_COLUMNS = ('neighbours', 'resonant_argument', 'cause_role', 'cause_term', 'commensurability')


def _accepted(key, regime=owner.REGIME_LIBRATION):
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
        reason='accepted:r2',
        regime=regime,
        route='r2',
    )


def _owner(intervals, windows, key='17E-36+0+19', regime=owner.REGIME_LIBRATION, step_yr=NAN):
    """An accepted owner measured on the episode `windows` (id, t0, t1, period) as the analysis does
    (`owner.cover_episodes`); the decisive episode is the last window."""
    candidate = _accepted(key, regime)
    detail = OriginDetail.OWNER_LIBRATION if regime == owner.REGIME_LIBRATION else OriginDetail.OWNER_FORCED_WAVE
    result = make(
        OwnerResult,
        n_candidates=1,
        candidates=[candidate],
        accepted=candidate,
        check=CheckValue.FAIL,
        psi_rate_rad_yr=0.0,
        detail=str(detail),
        intervals=list(intervals),
        step_yr=step_yr,
    )
    _, t0, t1, period = windows[-1]
    result.decisive_window = (t0, t1)
    result.cover = owner.covered_fraction(result.intervals, (t0, t1))
    owner.cover_episodes(result, windows, period)
    return result


def _physics(*ids):
    return [make(EpisodePhysics, episode_id=i, a_band_std_before_au=0.1, a_band_std_in_au=1.0) for i in ids]


def _decide(owner_result, geometry, ids=(1,)):
    """decide_status of an episodic angle whose own force law and potential pass (the dynamic leaf without an owner)."""
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
    return decide.decide_status(
        EPISODIC, SAMPLING, geometry, energy, response, None, None, owner_result, None, S, Ledger.from_settings(S), episodes=_physics(*ids)
    )


def _roles(spec_key, decision, owner_result):
    return roles.verdict_roles(
        ResonanceSpec.from_key(spec_key), decision.origin, decision.origin_detail, decision.owner, decision.flags, owner_result, S
    )


# --------------------------------------------------------------------------------------------- commensurability, terms


@pytest.mark.parametrize(
    'key, expected',
    [
        ('9M-16+0+7', '9M-16'),
        ('9M-16+2+5', '9M-16'),  # the members of one multiplet share it (rule C: counted once)
        ('2M-6J-1+0+0+5', '2M-6J-1'),
        ('3M+7J-6+0+0-4', '3M+7J-6'),
        ('1E-1+0+0', '1E-1'),
        ('2J-1', '2J-1'),  # the short notation
        ('1J-3r0', '1J-3r'),  # retrograde: the key without k
    ],
)
def test_commensurability(key, expected):
    assert roles.commensurability(ResonanceSpec.from_key(key)) == expected


@pytest.mark.parametrize(
    'key, expected',
    [
        ('3M+7J-6+0+0-4', '-4varpi'),  # 285637 (card g05-3)
        ('7E-10M-6+0+0+9', '+9varpi'),  # 481085
        ('3J+9S-2+0+0-10', '-10varpi'),  # 41588 (card g04-7)
        ('4M-7+1+2', '+2varpi+1varpi_M'),  # a member with a planet's varpi: the whole secular part
        ('5J-2S-2+0+0-1', '-1varpi'),
    ],
)
def test_secular_term(key, expected):
    assert roles.secular_term(ResonanceSpec.from_key(key)) == expected


def test_secular_term_of_a_retrograde_angle_uses_its_own_elements():
    # 1J-3r0: sigma = 3 lambda* - 1 lambda_J - (1 + 3 - 0) varpi* + 0 Omega (Li et al. 2019 eq. 2)
    assert roles.secular_term(ResonanceSpec.from_key('1J-3r0')) == '-4varpi*'
    # 2J-5r1: p = -(2 + 5 - 2) = -5, v = 2
    assert roles.secular_term(ResonanceSpec.from_key('2J-5r1')) == '-5varpi*+2Omega'


def test_position_reads_the_midpoint_against_the_hull_of_the_own_episodes():
    windows = [(1000.0, 2500.0), (7000.0, 8500.0)]
    assert roles.position((0.0, 1200.0), windows) == 'before'  # overlaps the first episode by 200 yr, mostly before
    assert roles.position((3000.0, 6000.0), windows) == 'during'  # between two own episodes
    assert roles.position((8000.0, 9900.0), windows) == 'after'
    assert roles.position((0.0, 9000.0), windows) == 'during'
    assert roles.position((0.0, 100.0), []) == ''


# --------------------------------------------------------------------------------------------- the roles on decisions


def test_owner_outside_the_episode_is_a_neighbour_before_and_after():
    # card g06-3 A: the owner librates on [0, 3000] and [6000, 9000], sigma on [3000, 6000] (cover 0): F7 refuses it.
    # The decision keeps its key for the reader, the verdict's owner column is empty and the neighbour is written with
    # its two blocks and where they lie
    result = _owner([(0.0, 3000.0, 'libration'), (6000.0, 9000.0, 'libration')], [(1, 3000.0, 6000.0, 1000.0)])
    geometry = make_geometry(
        window=(0.0, 9000.0), runs=[make_episode(episode_id=1, tp_times_yr=[3000.0, 3500.0, 4000.0, 4500.0, 5000.0, 5500.0, 6000.0])]
    )
    decision = _decide(result, geometry)
    assert decision.status == Status.TRANSIENT and str(Flag.OWNER_OUTSIDE_EPISODE) in decision.flags
    got = _roles('8M-9+0+1', decision, result)
    assert got.owner == ''
    assert got.neighbours == '17E-36+0+19 0–3000 before;17E-36+0+19 6000–9000 after'
    assert (got.cause_role, got.cause_term, got.resonant_argument) == (CauseRole.NEIGHBOUR, '17E-36+0+19', '')
    assert got.commensurability == '8M-9'


def test_neighbour_times_are_rounded_to_the_step_and_touching_regimes_join():
    # a libration piece and a forced-wave piece that overlap are one block; times to the 0.5-yr step
    result = _owner(
        [(10.2, 2000.3, 'libration'), (1900.0, 2999.8, 'forced_wave'), (7000.1, 9000.0, 'libration')],
        [(1, 3000.0, 6000.0, 1000.0)],
        step_yr=0.5,
    )
    assert roles.neighbours_text(result, S) == '17E-36+0+19 10.0–3000.0 before;17E-36+0+19 7000.0–9000.0 after'


def test_member_accepted_is_the_resonant_argument_not_the_owner():
    # card g06-1 A (31935): one resonance 9:16; the row takes the member's status, the member is its argument
    member = SubresonanceCandidate(
        ResonanceSpec.from_key('9M-16+2+5'), 1, False, 0.1, 0.0, 0.0, 0.0, np.nan, 1.0, np.nan, np.nan, np.nan, 1, False, ''
    )
    scan = replace(sub._empty((), CheckValue.FAIL, []), candidates=[member], accepted=member, theory_measured=True)
    member_verdict = Verdict(ResonanceType.MMR, Status.TRANSIENT, AngleStatus.EPISODIC, confidence=Confidence.HIGH)
    decision = decide._foreign(EPISODIC, scan, None, S, Ledger.from_settings(S), member_verdict)
    assert (decision.status, decision.reason, decision.origin) == (
        Status.TRANSIENT,
        Reason.SUBRESONANCE_LIBRATES,
        LibrationOrigin.SUBRESONANCE,
    )
    got = _roles('9M-16+0+7', decision, None)
    assert (got.owner, got.resonant_argument, got.cause_role, got.cause_term) == ('', '9M-16+2+5', CauseRole.NONE, '')
    assert got.commensurability == '9M-16'


def test_kinematic_libration_names_its_varpi_term():
    # card g05-3 (285637 3M+7J-6+0+0-4): the cycles come from -4 varpi, a does not respond; no owner is looked for
    got = roles.verdict_roles(
        ResonanceSpec.from_key('3M+7J-6+0+0-4'), LibrationOrigin.KINEMATIC, OriginDetail.SECULAR_CYCLE, '', [], None, S
    )
    assert (got.owner, got.cause_role, got.cause_term, got.neighbours) == ('', CauseRole.KINEMATIC, '-4varpi', '')
    ripple = roles.verdict_roles(
        ResonanceSpec.from_key('3M+7J-6+0+0-4'), LibrationOrigin.KINEMATIC, OriginDetail.PLANET_RIPPLE, '', [], None, S
    )
    assert ripple.cause_term == '2J-5S'


def test_forced_wave_owner_is_a_driver():
    # card g04-5 A (385266 16N-27: 3N-5 circulates, its forced wave makes the cycles): driver, kept in owner
    windows = [(0, 1000.0, 2500.0, 1000.0), (1, 7000.0, 8500.0, 1000.0)]
    driver = _owner([(0.0, 9000.0, 'forced_wave')], windows, key='3N-5+0+2', regime=owner.REGIME_FORCED_POSITIVE)
    runs = [
        make_episode(episode_id=0, tp_times_yr=[1000.0, 1500.0, 2000.0, 2500.0], period_med_yr=1000.0, centre_estimates=2),
        make_episode(episode_id=1, tp_times_yr=[7000.0, 7500.0, 8000.0, 8500.0], period_med_yr=1000.0, centre_estimates=2),
    ]
    decision = _decide(driver, make_geometry(window=(0.0, 9000.0), runs=runs), ids=(0, 1))
    assert (decision.reason, decision.origin, decision.origin_detail) == (
        Reason.INDUCED_LIBRATION,
        LibrationOrigin.INDUCED,
        str(OriginDetail.OWNER_FORCED_WAVE),
    )
    got = _roles('16N-27+0+11', decision, driver)
    assert (got.owner, got.cause_role, got.cause_term, got.neighbours) == ('3N-5+0+2', CauseRole.DRIVER, '3N-5+0+2', '')


def test_librating_owner_that_covers_the_episode_is_the_owner():
    windows = [(1, 3000.0, 6000.0, 1000.0)]
    result = _owner([(0.0, 9000.0, 'libration')], windows)
    geometry = make_geometry(
        window=(0.0, 9000.0), runs=[make_episode(episode_id=1, tp_times_yr=[3000.0, 3500.0, 4000.0, 4500.0, 5000.0, 5500.0, 6000.0])]
    )
    decision = _decide(result, geometry)
    assert decision.origin == LibrationOrigin.INDUCED
    got = _roles('8M-9+0+1', decision, result)
    assert (got.owner, got.cause_role, got.neighbours) == ('17E-36+0+19', CauseRole.OWNER, '')


def test_the_decisive_episode_row_keeps_only_an_owner_and_the_argument():
    # episodes.csv: the decisive episode no longer names a neighbour as its owner (2000 SF8 before A3a)
    decision = decide.StatusDecision(
        Status.TRANSIENT, Reason.DYNAMIC_EPISODE, LibrationOrigin.DYNAMIC, '', '17E-36+0+19', [str(Flag.OWNER_OUTSIDE_EPISODE)]
    )
    neighbour = roles.Roles(neighbours='17E-36+0+19 0–3000 before', cause_role=CauseRole.NEIGHBOUR, cause_term='17E-36+0+19')
    (physics,) = [analyse._with_origin(p, decision, neighbour) for p in _physics(1)]
    assert (physics.libration_origin, physics.owner, physics.resonant_argument) == ('dynamic', '', '')
    member = roles.Roles(resonant_argument='9M-16+2+5')
    (physics,) = [analyse._with_origin(p, decision, member) for p in _physics(1)]
    assert physics.resonant_argument == '9M-16+2+5'


# --------------------------------------------------------------------------------------------- the CSV


def test_the_roles_round_trip_and_an_old_row_reads_empty():
    verdict = Verdict(
        ResonanceType.MMR,
        Status.TRANSIENT,
        AngleStatus.EPISODIC,
        neighbours='4J-1+0-3 6412–13563 after',
        cause_role=CauseRole.NEIGHBOUR,
        cause_term='4J-1+0-3',
        commensurability='1V-5',
    )
    row = verdict.to_flat_dict()
    columns = list(row)
    assert columns[columns.index('owner') + 1 : columns.index('owner') + 6] == list(ROLE_COLUMNS)
    back = Verdict.from_flat_dict(row)
    assert (back.neighbours, back.cause_role, back.cause_term, back.commensurability, back.owner) == (
        '4J-1+0-3 6412–13563 after',
        CauseRole.NEIGHBOUR,
        '4J-1+0-3',
        '1V-5',
        '',
    )
    # a CSV read gives NaN cells: they restore as ''
    frame = pd.DataFrame([{**row, 'neighbours': None, 'cause_term': None}])
    restored = Verdict.from_flat_dict(frame.iloc[0].to_dict())
    assert restored.neighbours == '' and restored.cause_term == ''


# --------------------------------------------------------------------------------------------- benchmark (local data)


def _analysed(example):
    if not (io.example_path(example) / 'series.csv.gz').exists():
        pytest.skip('benchmark data not available')
    return analyse_mmr(io.load_series(example), S)


# (example, status, owner, neighbour key and position, resonant_argument, cause_role, cause_term): the author's
# answers of 2026-10-31 (g04-1 SF8, g06-3 GL4, g06-1 31935, g06-2 12218, g04-5 385266, g05-3 285637, g04-4 1862)
ROLE_ROWS = [
    ('mmr2__2000_SF8__8M-9+0+1', 'transient', '', ('17E-36+0+19', ('before', 'after')), '', 'neighbour', '17E-36+0+19'),
    ('mmr2__1999_GL4__1V-5+0+4', 'transient', '', ('4J-1+0-3', ('after',)), '', 'neighbour', '4J-1+0-3'),
    ('mmr3__1862__2M-6J-1+0+0+5', 'transient', '', ('6E-11+0+5', ('after',)), '', 'neighbour', '6E-11+0+5'),
    ('mmr2__31935__9M-16+0+7', 'transient', '', None, '9M-16+2+5', '', ''),
    ('mmr2__12218__4M-7+0+3', 'transient', '', None, '4M-7+1+2', '', ''),
    ('mmr2__385266__16N-27+0+11', 'non_resonant', '3N-5+0+2', None, '', 'driver', '3N-5+0+2'),
    ('mmr3__285637__3M+7J-6+0+0-4', 'non_resonant', '', None, '', 'kinematic', '-4varpi'),
]


@pytest.mark.benchmark
@pytest.mark.parametrize('example, status, owner_key, neighbour, argument, role, term', ROLE_ROWS)
def test_benchmark_roles(example, status, owner_key, neighbour, argument, role, term):
    verdict = _analysed(example).verdict
    assert str(verdict.status) == status
    assert (verdict.owner, verdict.resonant_argument, str(verdict.cause_role), verdict.cause_term) == (owner_key, argument, role, term)
    if neighbour is None:
        assert verdict.neighbours == ''
        return
    key, places = neighbour
    items = verdict.neighbours.split(';')
    assert all(item.startswith(f'{key} ') for item in items)
    assert set(places) <= {item.rsplit(' ', 1)[1] for item in items}
    assert not math.isnan(float(items[0].split(' ')[1].split('–')[0]))
