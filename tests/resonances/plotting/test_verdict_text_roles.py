"""W28 A3b: the roles of the other resonance in the header, the steps and the text (cards g04-4, g04-5, g04-6, g04-7,
g05-1, g05-3, g05-5, g06-1, g06-2, g06-4), and the hatching of induced episodes.

Fast tests build real result objects (the 2:1 kinematic synthetic of test_verdict_text, `OwnerResult`, `EpisodeCover`,
`SubresonanceResult`) and compare with strings worked out by hand from the keys and numbers given; the benchmark tests
read 719 36J-13, 385266 16N-27 and 41588 3J+9S-2."""

from __future__ import annotations

import math
from dataclasses import replace

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from resonances.plotting import mmr_combined as mc  # noqa: E402
from resonances.plotting import verdict_text as vt  # noqa: E402
from resonances.resonance.analysis.mmr import io  # noqa: E402
from resonances.resonance.analysis.mmr.models import (  # noqa: E402
    EpisodeCover,
    OwnerCandidate,
    OwnerResult,
    ResonanceSpec,
    SecondCapture,
    SubresonanceCandidate,
    SubresonanceResult,
)
from resonances.resonance.analysis.settings import settings_from_config  # noqa: E402
from resonances.resonance.analysis.verdict import CauseRole, CheckValue, LibrationOrigin, OriginDetail  # noqa: E402
from tests.resonances.analysis.builders import make, make_episode  # noqa: E402

S = settings_from_config()


# --------------------------------------------------------------------------------------------- the synthetic record


_SYNTHETIC_CACHE = {}


def _kinematic():
    """(analysis, row) of σ = 2λ_J − λ − ϖ with ϖ swinging ±0.9 rad in 400 yr and a flat a (test_verdict_text's 2:1
    kinematic synthetic): non_resonant / kinematic_libration, the A3a roles kinematic with the term −ϖ. Built once."""
    if 'analysis' not in _SYNTHETIC_CACHE:
        from scipy.optimize import brentq

        from resonances.data import const
        from resonances.resonance.analysis.mmr import context as ctx
        from resonances.resonance.analysis.mmr.analyse import analyse_mmr
        from resonances.resonance.analysis.mmr.models import MMRSeries, Regime

        spec = ResonanceSpec.from_key('2J-1+0-1')
        t = np.arange(0.0, 3000.0 + 0.25, 0.5)
        n_jupiter = const.PLANET_N_DEG_PER_GAUSS_YR['Jupiter'] * math.pi / 180
        a_au = brentq(lambda a: float(ctx.mean_motion(a)) - 2 * n_jupiter, 2.0, 5.0)
        lam = (2 * n_jupiter + 2 * math.pi / 60000.0) * t + 0.3
        varpi = 0.4 + 0.9 * np.sin(2 * math.pi * t / 400.0)
        sigma = 2 * (n_jupiter * t + 1.1) - lam - varpi
        ones = np.ones_like(t)
        a = a_au + 1e-6 * np.sin(2 * math.pi * t / 7.3)
        series = MMRSeries(
            t, a, 0.1 * ones, 0.05 * ones, 0.7 * ones, np.mod(varpi - 0.7, 2 * math.pi), np.mod(varpi, 2 * math.pi),
            np.mod(sigma, 2 * math.pi), spec, Regime(0.5, float(t[-1]), len(t)), {}, np.mod(lam, 2 * math.pi), name='a3b',
        )  # fmt: skip
        _SYNTHETIC_CACHE['analysis'] = analyse_mmr(series, S, scan_owner=False, scan_subresonances=False)
    analysis = _SYNTHETIC_CACHE['analysis']
    return analysis, {**analysis.numbers_row(), **analysis.summary_row()}


def _facts(analysis, row):
    return vt.MMRFacts(analysis, S, row, analysis.episode_rows())


def _with_owner(key: str, role: CauseRole, cover: float, detail: str = '', intervals=((1000.0, 2000.0, 'libration'),), **row):
    """The synthetic with an accepted owner `key` that the verdict names in `role` (A3a), its measured regimes
    `intervals`, its F7 cover of σ's window `cover`; extra row columns `row`."""
    analysis, base = _kinematic()
    accepted = make(OwnerCandidate, spec=ResonanceSpec.from_key(key), r2_median=0.94, accepted=True)
    window = EpisodeCover(None, 0.0, 3000.0, 400.0, cover)
    owner = make(OwnerResult, check=CheckValue.FAIL, accepted=accepted, n_candidates=1, candidates=[accepted],
                 intervals=list(intervals), episode_covers=(window,), cover=cover, detail=detail)  # fmt: skip
    verdict = replace(
        analysis.verdict, cause_role=role, owner=key, cause_term=key, libration_origin=LibrationOrigin.INDUCED,
        origin_detail=detail, reason='induced_libration', flags=[],
    )  # fmt: skip
    return _facts(replace(analysis, owner=owner, verdict=verdict), {**base, **row})


def _with_neighbour(neighbours: str, intervals, flags=('owner_outside_episode',)):
    """The synthetic with an owner F7 refused (A1/A3a): verdict role neighbour with `neighbours`, cover 0."""
    analysis, base = _kinematic()
    key = neighbours.split(' ')[0]
    accepted = make(OwnerCandidate, spec=ResonanceSpec.from_key(key), r2_median=0.99, accepted=True)
    window = EpisodeCover(None, 11228.0, 11683.0, 185.0, 0.0)
    owner = make(OwnerResult, check=CheckValue.FAIL, accepted=accepted, n_candidates=1, candidates=[accepted],
                 intervals=list(intervals), episode_covers=(window,), cover=0.0)  # fmt: skip
    verdict = replace(
        analysis.verdict, cause_role=CauseRole.NEIGHBOUR, owner='', cause_term=key, neighbours=neighbours,
        libration_origin=LibrationOrigin.UNKNOWN, origin_detail='', reason='geometric_episode', flags=list(flags),
        status='transient',
    )  # fmt: skip
    return _facts(replace(analysis, owner=owner, verdict=verdict), base)


# --------------------------------------------------------------------------------------------- words


@pytest.mark.parametrize(
    'term, expected',
    [
        ('-4varpi', '−4ϖ'),  # 285637 3M+7J-6+0+0-4 (card g05-3)
        ('-10varpi', '−10ϖ'),  # 41588 3J+9S-2+0+0-10
        ('+2varpi+1varpi_M', '2ϖ + ϖ_M'),
        ('-4varpi*', '−4ϖ*'),  # a retrograde angle's own ϖ*
        ('+1Omega-1Omega_J', 'Ω − Ω_J'),
        ('2J-5S', '2J-5S'),  # a planet ripple: a key, unchanged
    ],
)
def test_term_text(term, expected):
    assert vt.term_text(term) == expected


def test_argument_offset_of_rule_c():
    """31935 (card g06-1): 9M-16+2+5 − 9M-16+0+7 = (2 − 0) ϖ_M + (5 − 7) ϖ = −2ϖ + 2ϖ_M; 12218 (g06-2): 4M-7+1+2 −
    4M-7+0+3 = ϖ_M − ϖ; another commensurability has no offset."""
    spec = ResonanceSpec.from_key
    assert vt.argument_offset(spec('9M-16+0+7'), spec('9M-16+2+5')) == '−2ϖ + 2ϖ_M'
    assert vt.argument_offset(spec('4M-7+0+3'), spec('4M-7+1+2')) == '−ϖ + ϖ_M'
    assert vt.argument_offset(spec('4M-7+0+3'), spec('3M-5+0+2')) == ''


# --------------------------------------------------------------------------------------------- cause line and steps


def test_a_forced_wave_owner_is_a_driver():
    """Card g04-6 A / g04-5 (385266): origin_detail owner_forced_wave → 'driver', the period of the driver (4507.19 yr
    = 4.51 kyr), never 'owner:'; the step names the driver and its forced wave."""
    f = _with_owner('3N-5+0+2', CauseRole.DRIVER, 0.8, OriginDetail.OWNER_FORCED_WAVE, own_owner_period_yr=4507.19)
    role, text = vt.cause_line(f.v, f)
    assert (role, text) == ('driver', 'driver: 3N-5+0+2 (circulates, P 4.51 kyr): forced wave')
    assert 'owner:' not in text and vt.subtitle_role(f) == 'driver 3N-5+0+2'
    step = vt._owner_step(f)
    assert step.state == vt.StepState.LEAF
    assert step.outcome == 'driver 3N-5+0+2 circulates (P 4.51 kyr) and its forced wave drives a (R² 0.94) → non_resonant, induced'


def test_a_librating_owner_and_an_unconfirmed_one():
    """The owner librates in the episode (HW7 5E-16+0+11, cover 0.8 ≥ 0.5); with a cover of 0.3 < 0.5 the text does not
    confirm it: 'candidate driver' in the header, the subtitle and the step, never 'owner:'."""
    f = _with_owner('5E-16+0+11', CauseRole.OWNER, 0.8, OriginDetail.OWNER_LIBRATION)
    assert vt.cause_line(f.v, f) == ('owner', 'owner: 5E-16+0+11 (librates in the episode)')
    assert vt._owner_step(f).outcome.startswith('owner 5E-16+0+11 librates in σ’s episode and drives a (R² 0.94)')
    f = _with_owner('5E-16+0+11', CauseRole.OWNER, 0.3, OriginDetail.OWNER_LIBRATION)
    role, text = vt.cause_line(f.v, f)
    assert role == 'candidate driver' and text.startswith('candidate driver: 5E-16+0+11') and 'owner:' not in text
    assert vt.subtitle_role(f) == 'candidate driver 5E-16+0+11'
    assert vt._owner_step(f).outcome.startswith('candidate driver 5E-16+0+11 (R² 0.94)')


def test_the_1862_control_names_a_neighbour_with_its_interval():
    """Card g04-4/g04-6 (1862 2M-6J-1, the F7 control): 6E-11+0+5 librates at 13 733–17 914 yr, after σ's episode
    (11.2–11.7 kyr), cover 0: the header says it librates after the episode with that interval, outside it; the step
    calls it a neighbour, not the cause, and never 'owner:' or 'drives a'."""
    f = _with_neighbour('6E-11+0+5 13733–17914 after', [(13733.0, 17914.0, 'libration')])
    role, text = vt.cause_line(f.v, f)
    assert (role, text) == ('neighbour', '6E-11+0+5 librates after the episode (13.7 kyr–17.9 kyr), outside it')
    step = vt._owner_step(f)
    assert step.state == vt.StepState.PASSED and step.outcome.startswith('neighbour 6E-11+0+5 (R² 0.99) librates after')
    assert step.outcome.endswith('not the cause, the tree goes on') and 'covers 0 < 0.5' in step.outcome
    words = ' '.join((text, step.outcome, vt.driver_line(f)[0]))
    assert 'owner:' not in words and 'drives a' not in words
    assert vt.driver_line(f)[0].startswith('Candidate driver: 6E-11+0+5 (librates after the episode (13.7 kyr–17.9 kyr)')


def test_a_neighbour_before_and_after_and_during():
    """SF8 8M-9 (card g04-1): 17E-36+0+19 before and after; a block during the row's own episodes does not make σ's
    cycles; a forced-wave block 'drives a forced wave'."""
    f = _with_neighbour(
        '17E-36+0+19 1905–5236 before;17E-36+0+19 13558–16884 after', [(1905.0, 5236.0, 'libration'), (13558.0, 16884.0, 'libration')]
    )
    assert vt.neighbour_text(f) == '17E-36+0+19 librates before and after the episode (1.91 kyr–5.24 kyr, 13.6 kyr–16.9 kyr), outside it'
    f = _with_neighbour('1V-5+0+4 3000–5000 during', [(3000.0, 5000.0, 'forced_wave')], flags=('owner_psi_too_fast',))
    assert vt.neighbour_text(f) == '1V-5+0+4 drives a forced wave during 3 kyr–5 kyr but does not make σ’s cycles'


def test_the_kinematic_term_takes_the_owner_slot():
    """Cards g04-7 A and g05-3 (the author: the −4ϖ in the owner field): the synthetic is kinematic with σ's term −ϖ
    (σ = 2λ_J − λ − ϖ), B1 failed: 'kinematic: −ϖ term; a does not follow'; the owner step is n/a (no force on a).
    With origin_detail secular_cycle the period σ matched is named (2ω cycle 410 yr against a 400-yr swing)."""
    analysis, row = _kinematic()
    f = _facts(analysis, row)
    assert str(f.v.cause_role) == 'kinematic' and f.v.cause_term == '-1varpi'
    assert vt.cause_line(f.v, f) == ('kinematic', 'kinematic: −ϖ term; a does not follow')
    assert vt.subtitle_role(f) == 'kinematic −ϖ'
    (owner_step,) = [step for step in vt.mmr_steps(f) if step.key == 'owner']
    assert owner_step.state == vt.StepState.NOT_NEEDED and owner_step.outcome.startswith('n/a')
    period = float(f.episodes.decisive['period_med_yr'])
    assert abs(period / 400.0 - 1) < 0.1
    secular = vt.MMRFacts(
        replace(analysis, verdict=replace(analysis.verdict, origin_detail=OriginDetail.SECULAR_CYCLE)),
        S,
        {**row, 'kin_two_omega_period_yr': 410.0, 'kin_modulation_period_kyr': 50.0},
        analysis.episode_rows(),
    )
    assert vt.cause_line(secular.v, secular) == ('kinematic', 'kinematic: −ϖ term (2ω cycle, 410 yr); a does not follow')


def test_rule_c_names_the_resonant_argument():
    """Cards g06-1/g06-2 A: one resonance; the argument instead of an owner, in the header, the subtitle and the cause
    slot with the term by which the two angles differ."""
    analysis, row = _kinematic()
    spec = ResonanceSpec.from_key('9M-16+0+7')
    member = make(SubresonanceCandidate, spec=ResonanceSpec.from_key('9M-16+2+5'), accepted=True, reason='')
    sub = make(SubresonanceResult, candidates=[member], accepted=member, check=CheckValue.FAIL)
    verdict = replace(
        analysis.verdict, cause_role=CauseRole.NONE, resonant_argument='9M-16+2+5', commensurability='9M-16',
        libration_origin=LibrationOrigin.SUBRESONANCE,
    )  # fmt: skip
    f = _facts(replace(analysis, resonance=spec, subresonances=sub, verdict=verdict), row)
    assert vt.cause_line(f.v, f) == ('resonant argument', 'resonant argument: 9M-16+2+5 (one resonance 9M-16)')
    assert vt.subtitle_role(f) == 'resonant argument 9M-16+2+5'
    assert vt.argument_line(f) == (
        'Resonant argument: 9M-16+2+5, one resonance 9M-16 with this angle; the two differ by −2ϖ + 2ϖ_M, so this '
        'angle’s centre turns with that term.'
    )


def test_a_weak_member_is_a_frame_not_a_force():
    """Card g06-4 (2003 UG22 10E-17+6+1): a member test E′ rejected for its term (1e-9 of the strongest) is named as a
    frame, not a force; nothing when a member was accepted or none was rejected on strength."""
    analysis, row = _kinematic()
    weak = make(SubresonanceCandidate, spec=ResonanceSpec.from_key('10E-17+6+1'), reason='E_strength', coefficient_rel=1e-9)
    other = make(SubresonanceCandidate, spec=ResonanceSpec.from_key('10E-17+1+6'), reason='E_strength', coefficient_rel=3e-4)
    sub = make(SubresonanceResult, candidates=[other, weak], check=CheckValue.PASS)
    text = vt.frame_line(_facts(replace(analysis, subresonances=sub), row))
    assert text.startswith('A steadier centre of a member such as 10E-17+6+1 would be a frame, not a force: its term is 1e-9')
    assert 'own e, i, ω' in text
    assert vt.frame_line(_facts(replace(analysis, subresonances=replace(sub, candidates=[])), row)) == ''


# --------------------------------------------------------------------------------------------- both resonances at once?


def _capture(route: str, verdict: str, **columns):
    """The synthetic with an accepted owner (Ψ = 8σ − φ, HW7) whose second capture `verdict` was tested on 630 yr –
    19 kyr, and the capture columns `columns`."""
    f = _with_owner('5E-16+0+11', CauseRole.OWNER, 0.9, OriginDetail.OWNER_LIBRATION)
    capture = SecondCapture(0.338, math.radians(43.5), 0.251, float('nan'), verdict, 630.0, 19000.0, route=route)
    f.a = replace(f.a, owner=replace(f.a.owner, capture=capture))
    f.row = {**f.row, 'own_psi_def': '8*sigma-1*phi', 'own_second_capture': verdict, 'own_second_capture_route': route, **columns}
    return f


def test_both_resonances_at_once_not_ruled_out():
    """Card g05-5 A, HW7: x moved 43.5° < 180°, ẍ follows the secular harmonics (R² 0.883 ≥ 0.7) but also sin x
    (0.629 ≥ 0.2): not ruled out; the label is the question, never a count of captures."""
    f = _capture('', 'unresolved', own_psi_sweep_deg=43.5, own_psi_energy_R2=0.251, own_psi_secular_R2=0.883, own_psi_self_R2=0.629)
    (row,) = vt._capture_rows(f, vt._Printer(f.row))
    assert row.label == 'both resonances at once?' and row.icon == '?'
    assert row.text == (
        'x = Ψ/u, Ψ = 8*sigma-1*phi, on 630 yr–19 kyr; x moved 43.5° < 180°; energy fit R² 0.251 < 0.7; ẍ follows the '
        'secular harmonics (R² 0.883 ≥ 0.7), but also a well of its own (R² on sin x 0.629 ≥ 0.2) → not ruled out'
    )
    assert 'capture' not in row.text and 'captures' not in row.label


def test_both_resonances_at_once_excluded_by_the_secular_route():
    """78085 (A2, route F3): ẍ R² 0.983 ≥ 0.7 on the secular harmonics, 0.00106 < 0.2 on sin x → excluded."""
    f = _capture('secular_forcing', 'excluded', own_psi_sweep_deg=155.1, own_psi_secular_R2=0.983, own_psi_self_R2=0.00106)
    (row,) = vt._capture_rows(f, vt._Printer(f.row))
    assert row.icon == '✓' and row.text.endswith(
        'ẍ follows the secular harmonics (R² 0.983 ≥ 0.7), not a well of its own (R² on sin x 0.00106 < 0.2) → excluded'
    )


def test_hw7_identity_and_the_slow_swing():
    """Card g05-1: Ψ = 8σ − φ, so σ = φ/8 + Ψ/8 ('scaled by 1/8'), with the answer of 'both resonances at once?'; the
    slow swing of x at its level (A2's own_x_*): identified names the cycle, possible says 'probably' and that a slow
    libration of another resonance is not excluded, no cause says only that."""
    f = _capture(
        '', 'unresolved', own_x_period_yr=6600.0, own_x_halfamp_deg=11.5, own_x_secular_period_yr=7110.0, own_x_secular_term='2omega'
    )
    assert vt.identity_line(f).startswith('σ = φ/8 + Ψ/8: what φ (5E-16+0+11) does shows in σ scaled by 1/8;')
    assert vt.identity_line(f).endswith('(both resonances at once?: not ruled out).')
    head = 'σ’s centre swings against the owner with period 6.6 kyr (±11.5°)'
    f.row['own_x_cause_level'] = 'identified'
    assert vt.slow_x_line(f) == f'{head}: the 2ω cycle π/|ω̇| = 7.11 kyr.'
    f.row['own_x_cause_level'] = 'possible'
    assert vt.slow_x_line(f) == f'{head}: probably the 2ω cycle π/|ω̇| (7.11 kyr); a slow libration of another resonance is not excluded.'
    f.row['own_x_cause_level'] = ''
    assert vt.slow_x_line(f) == f'{head}: cause not identified; a slow libration of another resonance is not excluded.'


# --------------------------------------------------------------------------------------------- figure: header, hatch


def test_induced_episodes_are_hatched_apart_from_the_jump_hatch():
    """(a): an induced episode gets INDUCED_HATCH, distinct from EPISODE_BEFORE_JUMP_HATCH; both when both hold."""
    assert mc.INDUCED_HATCH != mc.EPISODE_BEFORE_JUMP_HATCH
    runs = [
        make_episode(episode_id=1, tp_times_yr=np.array([100.0, 250.0, 400.0])),
        make_episode(episode_id=2, tp_times_yr=np.array([500.0, 700.0, 900.0]), before_last_jump=True),
        make_episode(episode_id=3, tp_times_yr=np.array([950.0, 970.0, 990.0])),
    ]
    fig, ax = plt.subplots()
    mc.shade_runs(ax, runs, {1, 2, 3}, {}, lambda t: np.asarray(t), 1.0, mc.LAYOUT_V1, induced_ids={1, 2})
    hatches = [patch.get_hatch() for patch in ax.patches if patch.get_hatch()]
    plt.close(fig)
    assert sorted(hatches) == sorted([mc.INDUCED_HATCH, mc.EPISODE_BEFORE_JUMP_HATCH, mc.INDUCED_HATCH])


def test_induced_episode_ids_read_the_origin_the_analysis_wrote():
    analysis, _ = _kinematic()
    physics = [replace(p, libration_origin=str(LibrationOrigin.INDUCED)) for p in analysis.episode_physics]
    ids = {p.episode_id for p in physics if p.episode_id is not None}
    assert mc.induced_episode_ids(replace(analysis, episode_physics=physics)) == frozenset(ids)
    assert mc.induced_episode_ids(analysis) == frozenset()
    frozen = [{'episode_id': '2', 'libration_origin': 'induced'}, {'episode_id': '', 'libration_origin': 'induced'}]
    assert mc.induced_episode_ids(replace(analysis, frozen_episode_rows=frozen)) == frozenset({2})


def test_the_role_line_fits_the_header():
    """The header's second line holds the role; a long three-body neighbour stays inside the header's width and the
    record line moves under it."""
    analysis, _ = _kinematic()
    role = (
        '3J+9S-2+0+0-10 librates before and after the episode (1.91 kyr–5.24 kyr, 13.6 kyr–16.9 kyr, 21.1 kyr–28.4 kyr), '
        'outside it; drives a forced wave during 8.35 kyr–11.6 kyr but does not make σ’s cycles'
    )
    fig = plt.figure(figsize=mc.LAYOUT_V1.size, dpi=40)
    mc.draw_header(fig, analysis, mc.LAYOUT_V1, None, None, role)
    renderer = fig.canvas.get_renderer()
    (text,) = [t for t in fig.texts if t.get_text() == role]
    x, y, w, h = mc.LAYOUT_V1.header
    extent = text.get_window_extent(renderer)
    plt.close(fig)
    assert extent.x1 <= (x + w) * fig.dpi and extent.y0 >= y * fig.dpi


# --------------------------------------------------------------------------------------------- benchmark


ROLE_RECORDS = {
    'mmr2__719__36J-13+0-23': ('driver', 'driver: 11J-4+0-7 (circulates, P '),
    'mmr2__385266__16N-27+0+11': ('driver', 'driver: 3N-5+0+2 (circulates, P 4.5'),
    'mmr3__41588__3J+9S-2+0+0-10': ('kinematic', 'kinematic: −10ϖ term (2ω cycle, 11.3 kyr); a does not follow'),
}


@pytest.mark.benchmark
@pytest.mark.parametrize('example', sorted(ROLE_RECORDS))
def test_roles_on_the_benchmark(example):
    """Card g04-6 on 719 and 385266 (the driver, card g04-5: 3N-5+0+2 circulates with P 4.5 kyr), g04-7 on 41588
    (−10ϖ, its 10.5-kyr period matching the 2ω cycle 11.3 kyr): the role on the header's second line and in the
    subtitle, no 'owner:' for a driver; the induced episodes of 719 hatched on (a) and labelled on (l); overflow 0."""
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    series = io.load_series(example)
    analysis = analyse_mmr(series, S)
    row = {**analysis.numbers_row(), **analysis.summary_row()}
    f = vt.MMRFacts(analysis, S, row, analysis.episode_rows())
    role, start = ROLE_RECORDS[example]
    got_role, text = vt.cause_line(analysis.verdict, f)
    assert got_role == role and text.startswith(start), text
    assert 'owner:' not in text
    fig = mc.draw_combined(analysis, series, settings=S)
    try:
        texts = [t.get_text() for t in fig.texts]
        assert text in texts and fig.text_stats['overflow'] == 0
        if example.startswith('mmr2__719'):
            induced = mc.induced_episode_ids(analysis)
            assert induced and all(run.episode_id in induced for run in analysis.geometry.runs if run.accepted)
            hatched = [p for ax in fig.axes for p in ax.patches if p.get_hatch() == mc.INDUCED_HATCH]
            assert len(hatched) >= len(induced)
            labels = [t.get_text() for ax in fig.axes if ax.get_legend() for t in ax.get_legend().get_texts()]
            assert any(mc.INDUCED_LABEL in label for label in labels)
    finally:
        plt.close(fig)
