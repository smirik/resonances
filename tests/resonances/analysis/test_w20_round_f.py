"""W20 round F: the auto step avoids a step that folds a harmonic, the pilot's rejection of such a step names the
step to use, the harmonic flag survives G0, the figure text says what happened at step 5 and in the last regime,
and the round-E branches that had no test.

Independent numbers (by hand):
- P_orb = a^(3/2); a line m/P sampled at dt folds to |m/P - j/dt|, j = round(m dt / P), only when m dt / P >= 1/2.
  2J-1 with P_orb = 10.001 yr: min(P_orb, P_syn,J) / 2 = 5.0005 yr (P_syn,J = 1 / (1/10.001 - 1/11.862) = 63.7 yr),
  so the old auto step was 5 yr; there 2 x 5 / 10.001 = 0.9999 folds to 1 / |2/10.001 - 1/5| = 5 x 10.001 / 0.001 =
  50 005 yr > the 150-yr floor. At 2.5 yr 2 x 2.5 / 10.001 = 0.49995 < 1/2 and 2 x 2.5 / 63.7 = 0.08: nothing folds.
  P_orb = 10.1 yr: 5 yr folds to 1 / |2/10.1 - 1/5| = 505 yr, 2.5 yr nothing (0.495).
  P_orb = 2.005 yr: the bound 1.0025 yr gives 1 yr, which folds to 1 / |2/2.005 - 1| = 401 yr; at 0.5 yr
  2 x 0.5 / 2.005 = 0.4988 and P_syn,J = 1 / (1/2.005 - 1/11.862) = 2.413 yr gives 2 x 0.5 / 2.413 = 0.41: nothing.
- 1J-1 with P_orb = 12 yr at 5.999 yr: the second harmonic folds to 35 994 yr (round E's file); with
  MMR_FILTER_CUTOFF_YR = 10 yr the cutoff is not longer than two steps (10 <= 12.0): invalid / undersampled at G0.
- the 2J-1 pendulum below the Nyquist step (0.995 P_orb / 2) folds 2/P_orb to ~560 yr > 150 yr (round E's file).
"""

import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import resonances  # noqa: E402
from resonances.plotting import combined_numbers as numbers  # noqa: E402
from resonances.plotting import verdict_text as vt  # noqa: E402
from resonances.plotting.style import LAYOUT_V1  # noqa: E402
from resonances.resonance.analysis import orbit_type as ot  # noqa: E402
from resonances.resonance.analysis import pendulum  # noqa: E402
from resonances.resonance.analysis.ledger import Ledger  # noqa: E402
from resonances.resonance.analysis.mmr import context as context_module  # noqa: E402
from resonances.resonance.analysis.mmr import decide, io, response as response_module, sampling  # noqa: E402
from resonances.resonance.analysis.mmr.analyse import analyse_mmr, flag_folded_harmonic  # noqa: E402
from resonances.resonance.analysis.mmr.models import CheckValue, ResonanceSpec  # noqa: E402
from resonances.resonance.analysis.pendulum import EnergyAnalysis, SeparatrixMechanism  # noqa: E402
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, Flag, Reason  # noqa: E402
from resonances.data import const  # noqa: E402
from resonances.simulation import dry_run as dr  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from test_w20_round_e import S, _one_to_one_series, _pendulum_2j1, _run  # noqa: E402

from tests.resonances.analysis.builders import make_geometry  # noqa: E402

FLAGGED = str(Flag.SHORT_PERIOD_ALIASED)


def _simulation(period_yr, **config):
    sim = resonances.Simulation(
        name='round-f', date='2026-09-14', save=None, plot=None, save_summary=False, integration_years=100000, **config
    )
    sim.add_body(dict(a=period_yr ** (2 / 3), e=0.05, inc=0.05, Omega=1.0, omega=1.5, M=0.3), ['2J-1'], name='round-f')
    return sim


# --------------------------------------------------------------------------------------------- 1. the auto step


@pytest.mark.parametrize(
    'period_yr, bound_yr, old_auto, old_fold_yr, auto',
    [(10.001, 5.0005, 5.0, 50005.0, 2.5), (10.1, 5.05, 5.0, 505.0, 2.5), (2.005, 1.0025, 1.0, 401.0, 0.5)],
)
def test_the_auto_step_walks_down_to_a_step_that_folds_no_harmonic(period_yr, bound_yr, old_auto, old_fold_yr, auto):
    """Review E #1: the snapped short-period bound folds the second harmonic of P_orb far slower than the 150-yr floor;
    the auto step is the next round step below at which nothing folds, and the run's rows read '' there."""
    report = _simulation(period_yr, output_step='auto').dry_run('static')
    assert report.run['step_limits_yr']['short_period'] == pytest.approx(bound_yr, rel=1e-6)
    assert report.run['step_auto_yr'] == auto and report.run['binding_rule'] == dr.SHORT_PERIOD_HARMONIC_LIMIT
    row = report.rows.iloc[0]
    assert row.step_yr == auto and row.short_period_trigger == '' and row.expected_flags == ''
    settings = report.simulation.config.analysis_settings
    limits = dr._row_limits(
        ResonanceSpec.from_key('2J-1'),
        dr.static_inputs('round-f', report.simulation.bodies[0].initial_data, report.simulation.config.dry_run_settings)[0],
        settings,
        dr._planet_tables(),
    )
    assert limits['floor'] == pytest.approx(150.0)
    folded = sampling.folded_slowest_yr(limits['orbital_period'], limits['synodic_all'], old_auto, settings.mmr_short_period_harmonics)
    assert folded == pytest.approx(old_fold_yr, rel=2e-3)
    assert dr.folds_short_periods(limits, old_auto, settings) == sampling.SHORT_PERIOD_HARMONIC


def test_the_auto_step_of_a_run_without_a_harmonic_fold_is_unchanged():
    """The Hilda of the round-W4 test: the bound 3.95 yr gives 2.5 yr, where 2 x 2.5 / 7.905 = 0.63 folds to
    1 / |2/7.905 - 1/2.5| = 6.8 yr, faster than the floor: the binding rule stays short_period."""
    sim = resonances.Simulation(name='round-f', date='2026-09-14', save=None, plot=None, save_summary=False, integration_years=100000)
    sim.add_body(dict(a=3.9683768, e=0.1385, inc=0.1367, Omega=3.98, omega=0.68, M=2.42), ['3J-2+0-1'], name='153')
    report = sim.dry_run('static')
    assert report.run['step_auto_yr'] == 2.5 and report.run['binding_rule'] == 'short_period'
    assert report.run['step_fold_free_yr'] == 2.5


def _numeric_run():
    """2J-1 with P_orb = 10.001 yr at a configured 5-yr step: below the 5.0005-yr bound, the harmonic folds."""
    report = _simulation(10.001, output_step=5).dry_run('static')
    run = report.run
    steps = {'limits': run['step_limits_yr'], 'configured': run['step_configured_yr'], 'fold_free': run['step_fold_free_yr']}
    return report, steps


@pytest.mark.parametrize('pilot_ok, expected', [(False, {dr.SHORT_PERIOD_HARMONIC_LIMIT: 2.5}), (True, {}), (None, {})])
def test_a_rejected_harmonic_step_is_an_exceeded_limit_with_the_step_to_use(pilot_ok, expected):
    """Review E #1: the configured step is under every limit (5 < 5.0005, 7.5, ...) but folds a harmonic; the pilot's
    rejection makes it an exceeded 'short_period_harmonic' limit whose value is the fold-free step 2.5 yr; an
    acceptance or no measurement adds nothing."""
    report, steps = _numeric_run()
    rows = report.rows.to_dict('records')
    assert rows[0]['short_period_trigger'] == sampling.SHORT_PERIOD_HARMONIC and steps['fold_free'] == 2.5
    for row in rows:
        row['pilot_step_ok'] = pilot_ok
    assert dr._exceeded_limits(steps, rows) == expected


def test_the_pilot_note_names_the_fold_free_step():
    """The note of a rejected harmonic row: 'resample at 2.5 yr', not 'a step at which ...'."""
    sim = _simulation(10.001, output_step=5)
    config = sim.config
    settings, dry = config.analysis_settings, config.dry_run_settings
    tables = dr._planet_tables()
    body = sim.bodies[0]
    inputs, body_row = dr.static_inputs('round-f', body.initial_data, dry)
    plan = dr._plan(sim, [(body, inputs, body_row)], 100000.0, settings, dry, tables)
    (_, _, row) = plan['rows'][0]
    pilot = {
        'leak_deg': 3.0,
        'leak_adot_rel': 0.9,
        'leak_tp_prominence_deg': 1.0,
        'leak_slips': 0,
        'pilot_step_ok': False,
        'leak_step_yr': row['step_yr'],
        'forced_terms': '',
        'forced_lines': [],
    }
    dr._merge_pilot_rows(plan['rows'], {('round-f', row['resonance']): pilot}, settings, plan['steps']['fold_free'])
    (note,) = [n for n in row['pilot_notes'] if 'does not accept' in n]
    assert 'resample at 2.5 yr' in note
    dr._merge_pilot_rows(plan['rows'], {('round-f', row['resonance']): pilot}, settings, None)
    assert 'none of DRYRUN_AUTO_STEPS_YR is' in [n for n in row['pilot_notes'] if 'does not accept' in n][0]


# --------------------------------------------------------------------------------------------- 2. the flag at G0


def test_the_harmonic_flag_survives_the_g0_early_return():
    """Review E #3: with MMR_FILTER_CUTOFF_YR = 10 yr the record stops at G0 (invalid / undersampled), and now keeps
    the flag the dry-run predicts; no cap (the cap stays with the measurements). Without the setting: uncertain,
    flagged and capped, as before."""
    stopped = analyse_mmr(_one_to_one_series(5.999), replace(S, mmr_filter_cutoff_yr=10.0))
    v = stopped.verdict
    assert (str(v.status), str(v.reason)) == ('invalid', 'undersampled')
    assert stopped.sampling.folded_slowest_yr == pytest.approx(35994.0, rel=1e-6)
    assert FLAGGED in v.flags and v.caps == []
    measured = analyse_mmr(_one_to_one_series(5.999), S).verdict
    assert FLAGGED in measured.flags and FLAGGED in measured.caps
    assert measured.flags.count(FLAGGED) == 1


def test_the_early_flag_keeps_the_harmonic_ledger_entry():
    """The harmonic comparison is still written once by the cap (its title reads it)."""
    step = 0.995 * (math.pi / math.radians(const.PLANET_N_DEG_PER_GAUSS_YR['Jupiter'])) / 2
    result = analyse_mmr(_pendulum_2j1(step), S)
    entries = [e for e in result.verdict.decision_path if e.name.startswith('short_period_aliased: a harmonic')]
    assert len(entries) == 1 and entries[0].result is True
    assert result.verdict.flags.count(FLAGGED) == 1


# --------------------------------------------------------------------------------------------- 3. figure text


@pytest.mark.parametrize(
    'l4, r0, expected',
    [
        ('pass', 'pass', 'neither fails'),
        ('pass', 'not_measured', 'the force law decides'),
        ('not_measured', 'pass', 'the potential decides'),
        ('fail', 'pass', 'the potential carries it'),
        ('pass', 'fail', 'the force law carries it'),
        ('not_measured', 'not_measured', 'neither is measured'),
    ],
)
def test_step_five_says_which_test_passed_it(l4, r0, expected):
    """Verify E: 2024 WJ read 'potential not measured (…): neither fails'."""
    assert vt._force_conclusion(l4, r0) == expected


def test_step_five_of_1999_tj10_names_the_direct_strength():
    """Rule 5d/6 (i): L4 and R0 both not measured, the directly averaged strength passes (1.58 within [0.1, 10])."""
    example = 'mmr2__1999_TJ10__19M-22+0+3'
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    result = analyse_mmr(io.load_series(example), S)
    row = {**result.numbers_row(), **result.summary_row()}
    text = vt.explain_mmr(result, S, row, result.episode_rows())
    (step,) = [s for s in text.steps if s.key == 'force']
    ratio = result.strength.ratio
    assert 1 / S.mmr_strength_check_ratio <= ratio <= S.mmr_strength_check_ratio
    assert step.outcome.endswith(
        f'neither is measured; the strength by direct averaging decides (required / theory {vt.fmt(ratio)} within '
        f'[{vt.fmt(1 / S.mmr_strength_check_ratio)}, {vt.fmt(S.mmr_strength_check_ratio)}]) → the angle’s own dynamics'
    )


def test_wj_step_five_names_the_failed_energy_integral():
    example = 'coorbital__2024_WJ__1E-1+0+0'
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    result = analyse_mmr(io.load_series(example), S)
    row = {**result.numbers_row(), **result.summary_row()}
    text = vt.explain_mmr(result, S, row, result.episode_rows())
    (step,) = [s for s in text.steps if s.key == 'force']
    assert step.outcome.startswith('the force law holds; the potential is not measured (the 1:1 energy integral fails')
    assert step.outcome.endswith(': the force law decides → the angle’s own dynamics') and 'neither fails' not in step.outcome


@pytest.mark.parametrize(
    'regime, phrase',
    [
        ('circulation', 'σ circulates'),
        ('libration', 'σ librates'),
        ('drift', 'σ drifts'),
        ('insufficient_time', 'the record is too short to tell what σ does'),
        ('near_separatrix', 'σ moves near the separatrix'),
        ('pattern_x', 'σ is in the regime pattern x'),
    ],
)
def test_a_regime_in_a_sentence_is_plain_words(regime, phrase):
    assert vt.regime_phrase(regime) == phrase
    assert '_' not in vt.regime_phrase(regime)


@pytest.mark.parametrize(
    'example, phrase',
    [
        ('mmr2__1999_TJ10__19M-22+0+3', 'to the end the record is too short to tell what σ does'),
        ('coorbital__536531__1M-1+0+0__20kyr', 'to the end the record is too short to tell what σ does'),
        ('mmr3__501710__1E+7S-4+0+0-4__500kyr', 'to the end σ drifts'),
    ],
)
def test_no_raw_regime_name_in_the_sentences(example, phrase):
    """Verify E: 'from 18.8 kyr to the end σ is in insufficient_time' (1999 TJ10 19M-22 and 536531)."""
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    result = analyse_mmr(io.load_series(example), S)
    row = {**result.numbers_row(), **result.summary_row()}
    text = vt.explain_mmr(result, S, row, result.episode_rows())
    joined = ' '.join(text.sentences)
    assert phrase in joined
    for value in AngleStatus:
        if '_' in str(value):
            assert str(value) not in joined


# --------------------------------------------------------------------------------------------- 4. round-E gaps


def test_the_harmonic_flag_alone_keeps_exit_and_capture_off():
    """Review E #6: a 2J-1 record below the Nyquist step whose only blocker is the harmonic flag."""
    step = 0.995 * (math.pi / math.radians(const.PLANET_N_DEG_PER_GAUSS_YR['Jupiter'])) / 2
    series = _pendulum_2j1(step)
    context = context_module.build_context(series, S)
    smp = sampling.analyse_sampling(series, context, S)
    assert FLAGGED not in smp.flags and step < smp.short_period_step_yr  # no Nyquist flag
    assert flag_folded_harmonic(series.resonance, context, smp, S) is True and FLAGGED in smp.flags
    energy = EnergyAnalysis(
        None, None, 1.0, 1.0, transitions=[(10.0, pendulum.EXIT), (20.0, pendulum.CAPTURE)], mechanism=SeparatrixMechanism.UNEXPLAINED
    )
    args = dict(
        spec=series.resonance,
        angle_status=AngleStatus.EPISODIC,
        energy=energy,
        checks={'check_mmr_a_response': CheckValue.PASS, 'check_mmr_potential': CheckValue.NOT_MEASURED},
        caps=[],
        reason=Reason.DYNAMIC_EPISODE,
        subtype=AngleSubtype.RECURRENT,
    )
    other = [flag for flag in smp.flags if flag != FLAGGED]
    assert decide.exit_and_capture(flags=other, ledger=Ledger.from_settings(S), **args)
    assert not decide.exit_and_capture(flags=list(smp.flags), ledger=Ledger.from_settings(S), **args)


def _overflowing_text():
    checks = [vt.CheckRow(f'M{i}', f'margin number {i}', vt.CheckState.PASS, 'detail ' * 12, 'need', vt.COUNTS_FULL) for i in range(14)]
    checks.append(vt.CheckRow('cap', 'cap short_period_aliased', vt.CheckState.FAIL, 'held by the step', counts='cap ≤ medium'))
    return vt.VerdictText('resonant', 'libration', 'medium', 'reason x', ['A sentence. ' * 30] * 3, [], checks, 'raw: …')


def test_the_grey_numbers_go_first_and_the_cap_and_decisive_lines_stay():
    """Review E #6: the same 700 numbers with 690 plain lines do not fit even compact in five columns (a count of what
    is left); as grey lines (tests skipped by design) they are dropped first (W23 F1: before any other change of the
    layout, the explanation stays whole) and the 10 bold (decisive) lines and the cap row stay."""
    bold = [numbers.NumberLine('numbers', f'decisive{i}', str(i), group='D', style=numbers.STYLE_BOLD) for i in range(10)]
    for style, overflows in (('', True), (numbers.STYLE_GREY, False)):
        rest = [numbers.NumberLine('numbers', f'context{i}', str(i), group='G', style=style) for i in range(690)]
        fig = plt.figure(figsize=LAYOUT_V1.size, dpi=40)
        stats = numbers.draw_explained(fig, _overflowing_text(), bold + rest, LAYOUT_V1)
        texts = [artist.get_text() for artist in fig.texts]
        plt.close(fig)
        assert stats['compact'] == overflows and (stats['overflow'] > 0) == overflows and stats['grey_dropped']
        assert stats['n_columns'] == (5 if overflows else 4)
        assert any('cap short_period_aliased' in t for t in texts)
        if not overflows:
            assert all(any(t.startswith(f'decisive{i} ') for t in texts) for i in range(10))
            assert not any(t.startswith('context') for t in texts)


@pytest.mark.parametrize(
    'folded, floor, expected',
    [
        (150.0, 150.0, ''),  # on the floor: not slower than it (strict >)
        (math.nextafter(150.0, math.inf), 150.0, sampling.SHORT_PERIOD_HARMONIC),
        (35994.0, float('nan'), ''),  # no floor known
    ],
)
def test_the_predicate_on_the_floor_and_without_one(folded, floor, expected):
    assert sampling.short_period_trigger(5.999, 6.0, folded, floor) == expected


def test_the_header_of_a_retrograde_two_two_and_of_a_three_body_angle():
    """|p| = |q| = 2 retrograde is no 1:1 angle (`law_planet` None) and has no side; a three-body angle has no side.
    1J-1r0 is the retrograde co-orbital."""
    retro = ResonanceSpec.from_key('2J-2r0')
    assert retro.retrograde and ot.preprocess.law_planet(retro) is None
    about_zero = make_geometry(runs=[_run(0, 0)])
    assert ot.mmr_orbit_type(retro, about_zero, 0, None, S, 'libration') == 'retrograde, symmetric libration about 0°'
    assert ot.mmr_orbit_type(retro, about_zero, 0, None, S, 'circulation') == 'retrograde'
    one = ResonanceSpec.from_key('1J-1r0')
    assert ot.mmr_orbit_type(one, about_zero, 0, None, S, 'libration') == 'retrograde co-orbital, libration about 0°'
    three = ResonanceSpec.from_key('4J-2S-1+0+0-1')
    about_180 = make_geometry(runs=[_run(0, 180)])
    assert ot.mmr_orbit_type(three, about_180, 0, None, S, 'episodic') == 'symmetric libration about 180°'
    assert ot.mmr_orbit_type(three, about_180, 0, None, S, 'drift') == ''


def test_with_coorbital_potential_without_energy_or_law():
    """Review E #6: no energy analysis -> None; a response without a law, an empty geometry, filter_too_slow or no
    response leave the harmonic R0 (the energy itself returned); a law gives a copy with its R0, the energy unchanged."""
    pooled = SimpleNamespace(coorbital=SimpleNamespace(potential_check=CheckValue.PASS))
    episodes = SimpleNamespace(episodes=[object()])
    assert response_module.with_coorbital_potential(None, pooled, episodes, False) is None
    energy = EnergyAnalysis(None, None, 1.0, 1.0, with_slow=CheckValue.FAIL)
    for response, geometry, too_slow in (
        (SimpleNamespace(coorbital=None), episodes, False),
        (pooled, SimpleNamespace(episodes=[]), False),
        (pooled, episodes, True),
        (None, episodes, False),
    ):
        assert response_module.with_coorbital_potential(energy, response, geometry, too_slow) is energy
    read = response_module.with_coorbital_potential(energy, pooled, episodes, False)
    assert read.with_slow == CheckValue.PASS and read.potential_law is pooled.coorbital
    assert energy.with_slow == CheckValue.FAIL and energy.potential_law is None


@pytest.mark.parametrize(
    'l4, r0, expected',
    [
        ('fail', 'fail', 'both fail'),
        ('fail', 'not_measured', 'the force law fails, the potential is not measured'),
        ('not_measured', 'fail', 'the potential fails, the force law is not measured'),
    ],
)
def test_step_five_never_calls_a_failed_test_unmeasured(l4, r0, expected):
    """Review F #3: a measured failure is not 'neither is measured'."""
    assert vt._force_conclusion(l4, r0) == expected


def test_117306_step_five_names_the_potential_that_decided():
    """Review F #1, revised by W21 Q9: R0 passed when step 5 was decided (V_E ~ V_F 0.667 >= 0.666, rule 5d/6). The
    filter guard now runs before the tree and compares the decisive cycle with its own counterpart at 2 P_c, so it no
    longer withdraws R0 afterwards: the published check is the pass the tree read, and the step says so."""
    example = 'mmr2__117306__11M-20+0+9'
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    result = analyse_mmr(io.load_series(example), S)
    entry = next(e for e in result.verdict.decision_path if e.name == '5 R0 V_E ~ V_F windows')
    assert entry.result is True and str(result.verdict.checks.get('check_mmr_potential', '')) == 'pass'
    row = {**result.numbers_row(), **result.summary_row()}
    text = vt.explain_mmr(result, S, row, result.episode_rows())
    (step,) = [s for s in text.steps if s.key == 'force']
    assert 'the potential holds' in step.outcome
    assert 'later withdrawn' not in step.outcome
    assert 'neither is measured' not in step.outcome
