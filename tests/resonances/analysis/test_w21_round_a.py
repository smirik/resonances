"""W21 round A: the cap short_period_aliased made actionable (the author's Q1 of 2026-10-23). Changes no status or
confidence: the record gets its fold-free output step (smp_fold_free_step_yr) and, when the cap holds the level down,
action=resample at that step; the figure names the step.

Independent numbers (by hand):
- a line of frequency m/P sampled at dt folds when m dt / P >= 1/2, to |m/P - j/dt|, j = round(m dt / P); the step
  folds short periods when dt >= min(P)/2 (Nyquist) or a fold with m <= 2 is slower than the period floor;
- P_orb 10.001 yr, no planet, floor 150 yr (docs/dry-run.md): 5 yr < 5.0005 yr, but 2 x 5 / 10.001 = 0.9999 folds
  2/P to 1 / |0.199980 - 0.2| = 50 kyr; 2.5 yr: 2 x 2.5 / 10.001 = 0.49995 < 1/2, nothing folds -> 2.5 yr;
- P_orb 1.0002 yr, floor 150 yr, from 5 yr: 2.5, 2, 1 yr are >= 0.5001 yr (Nyquist); 0.5 yr is below it but
  2 x 0.5 / 1.0002 = 0.9998 folds 2/P to 1 / |1.99960 - 2| = 2500 yr > 150 yr; 0.25 yr: 0.4999 < 1/2 -> 0.25 yr.
  The same with the floor off (0): 0.5 yr (the harmonic part needs a floor);
- 2J-1, P_orb = 5.9353 yr, P_syn,J = 1 / (1/5.9353 - 1/11.862) = 11.88 yr, floor 150 yr, from 5.0012 yr: 2.5 yr is
  below 2.968 yr; 2 x 2.5 / 5.9353 = 0.842 folds to 1 / |0.33697 - 0.4| = 15.9 yr < 150 yr; P_syn: 2 x 2.5 / 11.88 =
  0.42, not folded -> 2.5 yr;
- the level (confidence.apply_caps): raw high, caps {short_period_aliased} -> medium, without it high: binds;
  with transient_unexplained (also <= medium) beside it: the level stays medium without it, does not bind; raw certain
  with zone_not_measured (<= high): medium, without it high: binds; raw medium: medium either way.
"""

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from resonances.plotting import verdict_text as vt
from resonances.resonance.analysis.confidence import Cap, cap_binds
from resonances.resonance.analysis.mmr import io, sampling
from resonances.resonance.analysis.mmr.analyse import analyse_mmr, short_period_resample
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import Action, Flag
from tests.resonances.analysis.test_w20_round_e import N_J, _pendulum_2j1

S = settings_from_config()
CAP = str(Cap.SHORT_PERIOD_ALIASED)
STEP_2J1 = 0.995 * (math.pi / N_J) / 2  # round E's 2J-1 record: below min(P)/2, its second harmonic folds to ~562 yr


# --------------------------------------------------------------------------------------------- 1. the fold-free step


@pytest.mark.parametrize(
    'orbit, synodic, step, floor, grid, expected',
    [
        (10.001, {}, 5.0, 150.0, S.mmr_resample_steps_yr, 2.5),  # the harmonic fold at 5 yr, nothing at 2.5 yr
        (1.0002, {}, 5.0, 150.0, S.mmr_resample_steps_yr, 0.25),  # 0.5 yr: below Nyquist, its harmonic folds to 2500 yr
        (1.0002, {}, 5.0, 0.0, S.mmr_resample_steps_yr, 0.5),  # no floor: only the Nyquist part
        (5.9353, {'Jupiter': 11.88}, 5.0012, 150.0, S.mmr_resample_steps_yr, 2.5),
        (1.0002, {}, 5.0, 150.0, (0.5, 1.0, 2.0), math.nan),  # no grid step is free
        (12.0, {}, 1.0, 150.0, S.mmr_resample_steps_yr, math.nan),  # the record's step folds nothing
    ],
)
def test_the_fold_free_step_by_hand(orbit, synodic, step, floor, grid, expected):
    got = sampling.fold_free_step(orbit, synodic, step, floor, grid, S.mmr_short_period_harmonics)
    assert (math.isnan(got) and math.isnan(expected)) or got == expected


def test_the_fold_free_step_reads_the_analysis_predicate():
    """At the step returned the predicate is '', at every grid step between it and the record's step it is not."""
    orbit, step, floor = 1.0002, 5.0, 150.0
    found = sampling.fold_free_step(orbit, {}, step, floor, S.mmr_resample_steps_yr, 2)
    short = sampling.short_period_step(orbit, {})

    def trigger(dt):
        return sampling.short_period_trigger(dt, short, sampling.folded_slowest_yr(orbit, {}, dt, 2), floor)

    assert trigger(found) == ''
    assert [trigger(dt) for dt in S.mmr_resample_steps_yr if found < dt < step] == ['harmonic', 'nyquist', 'nyquist', 'nyquist']
    assert sampling.folded_slowest_yr(orbit, {}, 0.5, 2) == pytest.approx(2500.5, rel=1e-3)


def test_the_resample_grid_is_the_dry_run_grid_up_to_ten_years():
    """The analysis walks MMR_RESAMPLE_STEPS_YR, the run's auto step DRYRUN_AUTO_STEPS_YR: the same values <= 10 yr, so
    the two walks agree for every record step <= 20 yr (the dry-run's next value)."""
    from resonances.resonance.analysis.settings import AUTO_STEPS_DEFAULT

    assert tuple(S.mmr_resample_steps_yr) == tuple(step for step in AUTO_STEPS_DEFAULT if step <= max(S.mmr_resample_steps_yr))


# --------------------------------------------------------------------------------------------- 2. when the cap binds


@pytest.mark.parametrize(
    'confidence, raw, caps, expected',
    [
        ('medium', 'high', [CAP], True),
        ('medium', 'high', [CAP, 'transient_unexplained'], False),
        ('medium', 'certain', [CAP, 'zone_not_measured'], True),
        ('medium', 'medium', [CAP], False),
        ('high', 'high', ['zone_not_measured'], False),  # no such cap
        ('', '', [CAP], False),  # uncertain / invalid: no level
    ],
)
def test_the_cap_binds_by_hand(confidence, raw, caps, expected):
    assert cap_binds(confidence, raw, caps, Cap.SHORT_PERIOD_ALIASED) is expected


def _assessment(confidence, raw, caps):
    return SimpleNamespace(confidence=confidence, confidence_raw=raw, caps=caps)


def test_short_period_resample_keeps_other_actions():
    smp = SimpleNamespace(fold_free_step_yr=0.5)
    binding = _assessment('medium', 'high', [CAP])
    assert short_period_resample(Action.SPOT_CHECK, None, binding, smp) == (Action.RESAMPLE, 0.5)
    # an alias/undersampled resample keeps its own step
    assert short_period_resample(Action.RESAMPLE, 0.2, binding, smp) == (Action.RESAMPLE, 0.2)
    # another cap at the same ceiling: still a spot check
    both = _assessment('medium', 'high', [CAP, 'transient_unexplained'])
    assert short_period_resample(Action.SPOT_CHECK, None, both, smp) == (Action.SPOT_CHECK, None)
    # no fold-free step on the grid: the action stays
    assert short_period_resample(Action.SPOT_CHECK, None, binding, SimpleNamespace(fold_free_step_yr=math.nan)) == (
        Action.SPOT_CHECK,
        None,
    )


# --------------------------------------------------------------------------------------------- 3. end to end


def test_the_2j1_record_asks_for_its_fold_free_step():
    """Round E's 2J-1 pendulum below the Nyquist step: medium (raw high) only by short_period_aliased -> resample at
    2.5 yr, where the second harmonic folds to 15.9 yr; status and level unchanged."""
    result = analyse_mmr(_pendulum_2j1(STEP_2J1), S)
    v, smp = result.verdict, result.sampling
    assert (str(v.status), str(v.confidence), str(v.confidence_raw), list(v.caps)) == ('resonant', 'medium', 'high', [CAP])
    assert smp.fold_free_step_yr == 2.5
    assert (v.action, v.resample_step_yr) == (Action.RESAMPLE, 2.5)
    assert result.numbers_row()['smp_fold_free_step_yr'] == 2.5
    assert result.summary_row()['resample_step_yr'] == 2.5


def test_no_free_grid_step_keeps_the_spot_check():
    settings = settings_from_config(mmr_resample_steps_yr=(5.0, 10.0))  # nothing below the record's step
    result = analyse_mmr(_pendulum_2j1(STEP_2J1), settings)
    v = result.verdict
    assert math.isnan(result.sampling.fold_free_step_yr) and str(v.confidence) == 'medium'
    assert (v.action, v.resample_step_yr) == (Action.SPOT_CHECK, None)


def test_a_step_that_folds_nothing_has_no_fold_free_step():
    result = analyse_mmr(_pendulum_2j1(1.0, span_yr=6000.0), S)
    assert str(Flag.SHORT_PERIOD_ALIASED) not in result.verdict.flags
    assert math.isnan(result.sampling.fold_free_step_yr) and result.verdict.action != Action.RESAMPLE


# --------------------------------------------------------------------------------------------- 4. benchmark rows

# v2-w20f (benchmark/data/v2-w20f/summary.csv): status, confidence, raw, action before W21 A; the fold-free step of
# the dry-run's walk (`dry_run.fold_free_step`, cross-checked on these rows) after it
BENCHMARK = [
    ('mmr2__2001_SS287__2J-1+0-1', 'resonant', 'medium', 'high', 2.5),
    ('mmr2__1999_GL4__1V-5+0+4', 'transient', 'medium', 'high', 0.25),
    ('mmr2__2002_XB__5V-7+0+2', 'resonant', 'medium', 'high', 0.25),
]


@pytest.mark.benchmark
@pytest.mark.parametrize('example, status, confidence, raw, step', BENCHMARK)
def test_benchmark_rows_held_by_the_cap(example, status, confidence, raw, step):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    result = analyse_mmr(io.load_series(example), S)
    v = result.verdict
    assert (str(v.status), str(v.confidence), str(v.confidence_raw), list(v.caps)) == (status, confidence, raw, [CAP])
    assert (v.action, v.resample_step_yr) == (Action.RESAMPLE, step)
    assert result.sampling.fold_free_step_yr == step
    row = {**result.numbers_row(), **result.summary_row()}
    text = vt.explain_mmr(result, S, row, result.episode_rows())
    assert f'a rerun at ≤ {vt.fmt(step)} yr output step lifts this cap' in ' '.join(text.sentences)
    assert f'Action: rerun with an output step of {vt.fmt(step)} yr (it folds no short period' in text.sentences[-1]


@pytest.mark.benchmark
def test_benchmark_alias_resample_keeps_its_own_step():
    """1864 51M-50: invalid/alias resample at 0.2 yr (the alias bounds); its fold-free step 0.5 yr is reported only."""
    example = 'mmr2__1864__51M-50+0-1'
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    result = analyse_mmr(io.load_series(example), S)
    v = result.verdict
    assert (str(v.status), str(v.reason), v.action, v.resample_step_yr) == ('invalid', 'alias', Action.RESAMPLE, 0.2)
    assert result.sampling.fold_free_step_yr == 0.5


@pytest.mark.benchmark
def test_benchmark_row_with_a_second_cap_keeps_the_spot_check():
    """1221 3E-8: transient_unexplained holds the level at medium too, so the rerun would not raise it."""
    example = 'mmr2__1221__3E-8+0+5'
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    result = analyse_mmr(io.load_series(example), S)
    v = result.verdict
    assert set(v.caps) == {CAP, 'transient_unexplained'} and str(v.confidence) == 'medium'
    assert (v.action, v.resample_step_yr) == (Action.SPOT_CHECK, None)
    assert result.sampling.fold_free_step_yr == 0.5


# --------------------------------------------------------------------------------------------- 5. the figure text


def test_the_cap_row_names_the_step():
    default = vt.CAP_ALTERNATIVES[CAP][1]
    assert vt.short_period_look(0.25, default) == '(a): the fast spread about σ_f; a rerun at ≤ 0.25 yr output step lifts this cap'
    assert vt.short_period_look(math.nan, default) == default
    assert vt.short_period_look(None, default) == default


def test_the_action_line_names_the_step():
    held = SimpleNamespace(action='resample', resample_step_yr=2.5, confidence='medium', caps=[CAP])
    assert (
        vt.action_sentence(held)
        == 'Action: rerun with an output step of 2.5 yr (it folds no short period: lifts the cap short_period_aliased).'
    )
    alias = SimpleNamespace(action='resample', resample_step_yr=0.2, confidence='', caps=[])
    assert vt.action_sentence(alias) == 'Action: rerun with an output step of 0.2 yr.'


# ===================================================================================== Task B (Q9): the filter guard
# The x2 guard compares the episode the status reads with its counterpart at 2 P_c (largest time overlap), and runs
# before the tree: a passed R0 it withdraws is not_measured for the tree, a failed one stays an objection.
#
# Independent numbers (by hand, from the 117306 11M-20 columns of v2-w20f and both reviews of 23.10):
# - P_c: TPs 27460-28690 yr (14 ripple TPs), the episode 37552.9 / 39211.3 / 40940.5 yr (half-amplitude 161.88 deg,
#   period 3387.6 yr) and one TP after it: 18 TPs; 2 P_c: a ripple episode 27639.7 / 27731.8 / 27823.9 yr (0.805 deg,
#   184.2 yr), the episode 37536.7 / 39213.0 / 40942.3 yr (161.86 deg, 3405.6 yr), 2 more TPs: 8 TPs;
# - the old reading: 18 against 8 TPs (|10| > max(2, 0.1 x 18)), 1 against 2 episodes, 161.88 against 0.805 deg (the
#   first run with three TPs), the energy median over every cycle (0.95 + 0.98) / 2 = 0.965 against
#   (0.98 + 1.7) / 2 = 1.34 (across 1): all four change;
# - the pair: 3 against 3 TPs; starts 37552.9 - 37536.7 = 16.2 yr, ends 1.8 yr, both < 1 x 3387.6 yr; amplitudes
#   |161.86 - 161.88| = 0.02 deg < 0.2 x 161.88 = 32.4 deg; the episode's cycles 0.95, 0.98 (median 0.965) against
#   0.96, 0.98 (0.97): |0.005| < 0.1, both below 1: nothing changes; overlap 3387.6 / 3405.6 = 0.9947;
# - largest overlap: [100, 1000] against [100, 400] (300 yr) and [450, 1000] (550 yr) -> the second; against [0, 50]
#   and [2000, 3000] -> none (the episode is not reproduced).


def _run(times, amp_deg=0.0, period=float('nan'), episode_id=0, shift=0.0):
    times = np.asarray(times, dtype=float)
    return SimpleNamespace(
        episode_id=episode_id,
        tp_times_yr=times,
        t_start_yr=float(times[0]),
        t_end_yr=float(times[-1]),
        half_amp_max_rad=math.radians(amp_deg),
        period_med_yr=period,
        centre_shift_over_amp=shift,
    )


def _geometry(tp_times, episodes, period):
    return SimpleNamespace(
        window=(150.0, 49850.0),
        pattern_b=None,
        episodes=episodes,
        period_cycle_yr=period,
        turning_points=SimpleNamespace(times_yr=np.asarray(tp_times, dtype=float)),
    )


RIPPLE_X1 = list(np.linspace(27460.0, 28690.0, 14))
EPISODE_X1 = _run([37552.9, 39211.3, 40940.5], 161.88, 3387.6, episode_id=2)
EPISODE_X2 = _run([37536.7, 39213.0, 40942.3], 161.86, 3405.6, episode_id=1)
RIPPLE_X2 = _run([27639.7, 27731.8, 27823.9], 0.805, 184.2, episode_id=0)
X1 = _geometry(RIPPLE_X1 + list(EPISODE_X1.tp_times_yr) + [45000.0], [EPISODE_X1], 3387.6)
X2 = _geometry(list(RIPPLE_X2.tp_times_yr) + list(EPISODE_X2.tp_times_yr) + [20000.0, 45000.0], [RIPPLE_X2, EPISODE_X2], 1794.9)
# cycle energies (h, separatrix 1) by mid-time: two cycles outside the episode, two inside it (below the separatrix)
H_X1, T_X1 = np.array([0.9, 2.0, 0.95, 0.98]), np.array([10000.0, 28000.0, 38400.0, 40100.0])
H_X2, T_X2 = np.array([1.7, 1.8, 0.96, 0.98]), np.array([10000.0, 28000.0, 38400.0, 40100.0])
INTERVAL = (150.0, 49850.0)


def _pair(x2_episode):
    from resonances.resonance.analysis.mmr import preprocess

    span = (EPISODE_X1.t_start_yr, EPISODE_X1.t_end_yr)
    first = preprocess.decisive_quantities(X1, H_X1, 75.0, INTERVAL, T_X1, EPISODE_X1, span)
    second = preprocess.decisive_quantities(X2, H_X2, 150.0, INTERVAL, T_X2, x2_episode, span)
    return first, second


def test_b_the_counterpart_is_the_largest_overlap():
    from resonances.resonance.analysis.mmr.preprocess import counterpart

    assert counterpart(EPISODE_X1, [RIPPLE_X2, EPISODE_X2]) is EPISODE_X2
    wide, left, right = _run([100.0, 1000.0]), _run([100.0, 400.0]), _run([450.0, 1000.0])
    assert counterpart(wide, [left, right]) is right  # 550 yr against 300 yr
    assert counterpart(wide, [_run([0.0, 50.0]), _run([2000.0, 3000.0])]) is None


def test_b_the_record_wide_reading_compares_a_ripple_with_the_cycle():
    """The old reading (no decisive episode given): every quantity changes, the shape compares 161.88 with 0.805 deg."""
    from resonances.resonance.analysis.mmr import preprocess

    first = preprocess.decisive_quantities(X1, H_X1, 75.0, INTERVAL, T_X1)
    second = preprocess.decisive_quantities(X2, H_X2, 150.0, INTERVAL, T_X2)
    assert (first.n_tp, second.n_tp) == (18, 8)
    assert math.degrees(second.half_amp_max_rad) == pytest.approx(0.805)
    result = preprocess.compare_quantities(first, second, S)
    assert result.changed == ('turning_points', 'episodes', 'energy', 'shape') and result.sensitive is True


def test_b_the_decisive_pair_is_the_same_cycle():
    from resonances.resonance.analysis.mmr import preprocess

    first, second = _pair(EPISODE_X2)
    assert (first.n_tp, second.n_tp) == (3, 3)
    assert (first.h_med, second.h_med) == (pytest.approx(0.965), pytest.approx(0.97))
    assert (first.period_yr, second.period_yr) == (3387.6, 3405.6)
    assert second.n_episodes == 2  # the ripple episode is still counted, as a column
    result = replace(preprocess.compare_quantities(first, second, S), decisive_episode_id=2)
    assert result.changed == () and result.sensitive is False
    assert result.decisive_overlap == pytest.approx((40940.5 - 37552.9) / (40942.3 - 37536.7))
    columns = result.to_columns()
    assert columns['flt_x1_decisive_bounds_yr'] == '37552.9-40940.5' and columns['flt_x2_decisive_bounds_yr'] == '37536.7-40942.3'
    assert columns['flt_decisive_episode_id'] == 2


def test_b_an_episode_not_reproduced_is_a_change():
    from resonances.resonance.analysis.mmr import preprocess

    first, second = _pair(None)  # 2 P_c holds only the ripple episode: the cycle is not reproduced
    # the x2 TPs inside the span 37552.9-40940.5 yr: 39213.0 only (37536.7 and 40942.3 lie just outside), no shape
    assert second.n_tp == 1 and not np.isfinite(second.half_amp_max_rad)
    result = replace(preprocess.compare_quantities(first, second, S), decisive_episode_id=2)
    assert result.changed == ('turning_points', 'episodes', 'shape') and result.decisive_overlap == 0.0


def _energy(with_slow):
    from resonances.resonance.analysis.pendulum import EnergyAnalysis

    return EnergyAnalysis(None, None, 0.667, 1.0, with_slow=with_slow)


@pytest.mark.parametrize(
    'with_slow, changed, tree, withdrawn',
    [
        ('pass', ('energy',), 'not_measured', ('check_mmr_potential',)),  # a passed R0 the guard withdraws
        ('fail', ('energy',), 'fail', ()),  # a failure stays an objection
        ('pass', ('turning_points', 'shape'), 'pass', ()),  # R0 does not read these
        ('not_measured', ('energy',), 'not_measured', ()),
    ],
)
def test_b_energy_for_tree(with_slow, changed, tree, withdrawn):
    from resonances.resonance.analysis.ledger import Ledger
    from resonances.resonance.analysis.mmr import decide
    from resonances.resonance.analysis.mmr.preprocess import DecisiveQuantities, FilterSensitivity
    from resonances.resonance.analysis.verdict import CheckValue

    energy = _energy(CheckValue(with_slow))
    same = DecisiveQuantities(75.0, 3, 1, False, 0.0, 0.9)
    sensitivity = FilterSensitivity(same, same, changed)
    ledger = Ledger.from_settings(S)
    read, taken = decide.energy_for_tree(energy, sensitivity, ledger)
    assert str(read.with_slow) == tree and tuple(item.check for item in taken.values()) == withdrawn
    assert all(margin == 'R0' and item.guard == decide.WITHDRAWN_FILTER_GUARD for margin, item in taken.items())
    assert str(energy.with_slow) == with_slow  # the energy's own result (its columns, the margins) is kept
    entries = [e for e in ledger.entries if e.name.startswith('filter guard')]
    assert len(entries) == len(withdrawn) and all(e.result is None for e in entries)
    assert decide.energy_for_tree(energy, FilterSensitivity(same, None, ()), ledger) == (energy, {})  # not checked


BENCH_117306 = 'mmr2__117306__11M-20+0+9'


@pytest.mark.benchmark
def test_b_117306_the_guard_compares_the_cycle_with_itself():
    """The guard no longer fires on the 0.8-deg ripple at 27.6 kyr: the 161.9-deg cycle at 37.5-40.9 kyr is reproduced at
    2 P_c, R0 stays the pass the tree read (5d/6), and the verdict stays transient/episodic/medium (author: capture near
    40 kyr; both reviews: medium for one cycle at the separatrix)."""
    if not io.example_path(BENCH_117306).exists():
        pytest.skip('benchmark data not available')
    analysis = analyse_mmr(io.load_series(BENCH_117306), S)
    verdict, numbers = analysis.verdict, analysis.numbers_row()
    assert (str(verdict.status), str(verdict.angle_status), str(verdict.confidence)) == ('transient', 'episodic', 'medium')
    assert numbers['flt_filter_sensitive'] is False and numbers['flt_withdrawn_before_tree'] == ''
    assert numbers['flt_x1_decisive_bounds_yr'] == '37552.9-40940.5' and numbers['flt_x2_decisive_bounds_yr'] == '37536.7-40942.3'
    assert numbers['flt_x1_half_amp_max_deg'] == pytest.approx(161.88, abs=0.01)
    assert numbers['flt_x2_half_amp_max_deg'] == pytest.approx(161.86, abs=0.01)
    assert (numbers['flt_x1_n_tp'], numbers['flt_x2_n_tp'], numbers['flt_x2_n_episodes']) == (3, 3, 2)
    assert str(verdict.checks['check_mmr_potential']) == 'pass' and str(Flag.FILTER_SENSITIVE) not in verdict.flags
    assert next(e for e in verdict.decision_path if e.name == '5 R0 V_E ~ V_F windows').result is True


@pytest.mark.benchmark
def test_b_471155_a_withdrawn_r0_is_not_measured_for_the_tree():
    """471155 6N-7: R0 passes at P_c, the episode's cycle energy crosses h = 1 at 2 P_c (1.035 against 0.960): the guard
    withdraws R0 before the tree, which reads it as not measured (ledger value none); L4 passes, so the row stays
    transient/episodic/medium as in v2-w20f (where the guard withdrew it only afterwards)."""
    example = 'mmr2__471155__6N-7+0+1'
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    analysis = analyse_mmr(io.load_series(example), S)
    verdict, numbers = analysis.verdict, analysis.numbers_row()
    assert numbers['flt_withdrawn_before_tree'] == 'check_mmr_potential' and 'energy' in numbers['flt_sensitive_quantities']
    assert numbers['flt_x1_h_med'] > 1 > numbers['flt_x2_h_med']
    path = {entry.name: entry for entry in verdict.decision_path}
    assert path['5 R0 V_E ~ V_F windows'].value is None
    assert any(name.startswith('filter guard: check_mmr_potential passed at P_c') for name in path)
    assert str(verdict.checks['check_mmr_potential']) == 'not_measured' and str(verdict.checks['check_mmr_a_response']) == 'pass'
    assert (str(verdict.status), str(verdict.angle_status), str(verdict.confidence)) == ('transient', 'episodic', 'medium')


# ------------------------------------------------------------------- C. onset_failed only on a verdict claiming a capture
# docs/physics/response.md ("Onset of an episode (O1)"): graded only for a verdict that claims a capture (`decide.capture_claimed`: the
# angle's own, a subresonance's or the owner's libration). facts-a (2026-10-23) found the cap on 2897 35S-4
# non_resonant/implausible_strength: that verdict denies the angle's own libration and names no other.
# By hand (CAP_CEILING: implausible_strength, onset_failed <= low; transient_unexplained <= medium): raw medium with
# {implausible_strength, transient_unexplained} -> low (implausible_strength still binds); without implausible_strength
# -> medium (the facts-a capexp.py result once both low caps go); O1 = 1.03 >= 0.5 fails (MMR_O1_RATIO_FAIL).


@pytest.mark.parametrize(
    'status, origin, detail, claimed',
    [
        ('transient', 'dynamic', '', True),
        ('resonant', 'dynamic', '', True),
        ('transient', 'unknown', '', True),  # 5u: the geometry's own status claims the angle's libration
        ('non_resonant', 'unknown', '', False),  # 5s implausible_strength: no libration claimed
        ('non_resonant', '', '', False),
        ('transient', 'subresonance', 'composite_multiplet', True),
        ('non_resonant', 'induced', 'owner_libration', True),  # the owner's libration (1997 GK3 3M+5S-3)
        ('non_resonant', 'induced', 'owner_forced_wave', False),
        ('non_resonant', 'kinematic', 'varpi_term', False),
    ],
)
def test_c_capture_claimed_by_status_and_origin(status, origin, detail, claimed):
    from resonances.resonance.analysis.mmr import decide
    from resonances.resonance.analysis.verdict import LibrationOrigin, Reason, Status

    decision = decide.StatusDecision(Status(status), Reason.DYNAMIC_EPISODE, LibrationOrigin(origin), detail, '')
    assert decide.capture_claimed(decision) is claimed


@pytest.mark.parametrize(
    'status, origin, reason, capped',
    [
        ('transient', 'dynamic', 'dynamic_episode', True),
        ('non_resonant', 'unknown', 'implausible_strength', False),
        ('non_resonant', 'induced', 'induced_libration', True),
    ],
)
def test_c_collect_caps_follows_the_margin_predicate(status, origin, reason, capped):
    """A failed O1 (ratio 1.03 >= 0.5) handed to collect_caps: the cap only where the verdict claims a capture."""
    from resonances.resonance.analysis.mmr import decide
    from resonances.resonance.analysis.ledger import Margin
    from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, LibrationOrigin, Reason, Status

    detail = 'owner_libration' if origin == 'induced' else ''
    decision = decide.StatusDecision(Status(status), Reason(reason), LibrationOrigin(origin), detail, '')
    angle = decide.AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, 'G2', 0)
    margins = {'O1': decide._margin(Margin.FAIL, 1.03, 'O1')}
    caps = decide.collect_caps(decide.tree_view(angle, None, None), decision, None, margins, S)
    assert (str(Cap.ONSET_FAILED) in caps) is capped
    assert 1.03 >= S.mmr_o1_ratio_fail


def test_c_levels_by_hand():
    from resonances.resonance.analysis.confidence import Confidence, apply_caps

    medium = Confidence.MEDIUM
    assert apply_caps(medium, ['implausible_strength', 'transient_unexplained']) == Confidence.LOW
    assert apply_caps(medium, ['transient_unexplained']) == Confidence.MEDIUM
    assert apply_caps(Confidence.HIGH, ['onset_failed']) == Confidence.LOW


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'example, expected, onset',
    [
        # facts-a: non_resonant/implausible_strength, raw medium (L3 failed), O1 1.03 -> no cap, O1 not_applicable,
        # the level stays low by implausible_strength (the author decides that cap, not this round)
        ('mmr2__2897__35S-4+0-31', ('non_resonant', 'episodic', 'low', 'implausible_strength'), False),
        # a transient of the angle's own dynamics keeps the cap (O1 0.536 >= 0.5): low as in v2-w20f
        ('mmr2__34706__7J-2+0-5', ('transient', 'episodic', 'low', 'dynamic_episode'), True),
    ],
)
def test_c_benchmark_onset_cap(example, expected, onset):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    verdict = analyse_mmr(io.load_series(example), S).verdict
    got = (str(verdict.status), str(verdict.angle_status), str(verdict.confidence), str(verdict.reason))
    assert got == expected
    assert (str(Cap.ONSET_FAILED) in verdict.caps) is onset
    o1 = verdict.margins['O1']
    assert str(o1.margin) == ('fail' if onset else 'not_applicable')
    if not onset:
        assert str(Cap.IMPLAUSIBLE_STRENGTH) in verdict.caps and str(verdict.confidence_raw) == 'medium'


# ------------------------------------------------- D. O1's background holds no accepted episode of the angle (Q2)
# docs/physics/response.md ("Onset of an episode (O1)", `response.onset_group`): the 2P before the episode, but an accepted episode of the
# same angle ending there joins the episode's group when no full turn of sigma_f lies between them (the background
# is then the 2P before the group's first episode, P still the episode's own), and cuts the background at its end
# when one does; less than P left: not measured, 'episode'.
# By hand, P = 1000 yr, the episode R [10000, 15000]:
# - A [7000, 9500] (ends inside [8000, 10000]), no turn between: group (A), background [7000 - 2000, 7000] = [5000, 7000];
#   B [3000, 4800] ends before 5000: stays out; B [3000, 5500] ends inside: joins, background [1000, 3000];
# - a turn between A and R: cut at 9500, [9500, 10000] = 500 yr < P -> 'episode'; A' [7000, 8200]: [8200, 10000],
#   1800 yr >= P: measured;
# - two episodes A2 [8300, 9500] and A1 [7000, 8000]: A2 joins, then A1 ends inside [6300, 8300]: (A2, A1), [5000, 7000];
# - the regime's clear bound at 9600 (a jump): A ends before it and is not looked at, [8000, 10000] as before.
# Synthetic a (P = 500 yr): a wave of 0.0005 au before 6000 yr, 0.005 au in A [6000, 8500], the gap and R [9000, 14000]:
# the old background [8000, 9000] lies on A's wave, O1 = 1; the group's [5000, 6000] carries 0.0005 against 0.005,
# O1 = 0.1 (the band's gain cancels), plus the leak of A's onset through the band-pass (measured 0.16 < 0.25).
# Benchmark (v2-w20f -> W21 D; independent numbers: claude-q2 and codex-q2 of 23.10 measured O1 of #4+#5 = 0.73, 0.725):
# - 2013 VA14 60 kyr: #4 [40115, 41344] ends 393 yr before #5 (P 1720 yr) without a turn: O1 1.028 -> 0.733,
#   background [40115 - 3440, 40115] = [36675, 40115], still a fail (>= 0.5): uncertain/episodic as before;
# - 17445 5J-2S-2: #2 ends 6597 yr before #3 (P 6603 yr) with 716 deg of sigma_f between them: the cut leaves
#   6597 < 6603 yr, O1 1.019 (fail, onset_failed, low) -> not measured 'episode', transient/episodic/high;
# - 1863 7J-2: #0 ends 1130 yr before #1 (P 665 yr, 2P 1330 yr) with 1005 deg between them: background
#   [6567, 7697], O1 0.720 -> 0.456 (thin), transient/episodic low -> medium.

from resonances.resonance.analysis import geometry as geo  # noqa: E402
from resonances.resonance.analysis.mmr import response as rsp  # noqa: E402


def _ep(episode_id, start, end):
    return SimpleNamespace(episode_id=episode_id, t_start_yr=float(start), t_end_yr=float(end))


R_EP = _ep(9, 10000, 15000)


def _never(left, right):
    return False


def _always(left, right):
    return True


@pytest.mark.parametrize(
    'earlier, circulates, low, expected',
    [
        ([_ep(1, 7000, 9500), _ep(0, 3000, 4800)], _never, -np.inf, (5000.0, 7000.0, (1,), -np.inf)),
        ([_ep(1, 7000, 9500), _ep(0, 3000, 5500)], _never, -np.inf, (1000.0, 3000.0, (1, 0), -np.inf)),
        ([_ep(1, 7000, 9500)], _always, -np.inf, (9500.0, 10000.0, (), 9500.0)),
        ([_ep(1, 7000, 8200)], _always, -np.inf, (8200.0, 10000.0, (), 8200.0)),
        ([_ep(2, 8300, 9500), _ep(1, 7000, 8000)], _never, -np.inf, (5000.0, 7000.0, (2, 1), -np.inf)),
        ([_ep(1, 7000, 9500)], _never, 9600.0, (8000.0, 10000.0, (), -np.inf)),
        ([], _never, -np.inf, (8000.0, 10000.0, (), -np.inf)),
    ],
)
def test_d_onset_group_by_hand(earlier, circulates, low, expected):
    got = rsp.onset_group(R_EP, 1000.0, earlier + [R_EP], circulates, low)
    assert (got.start_yr, got.end_yr, got.group_ids, got.episode_cut_yr) == expected


def test_d_onset_clear_names_the_episode_cut():
    t = np.arange(0.0, 20001.0, 1.0)
    cut = rsp.onset_group(R_EP, 1000.0, [_ep(1, 7000, 9500), R_EP], _always)
    assert rsp.onset_clear(t, 1000.0, 10000.0, 15000.0, None, S, cut)[1] == rsp.ONSET_EPISODE
    kept = rsp.onset_group(R_EP, 1000.0, [_ep(1, 7000, 8200), R_EP], _always)
    assert rsp.onset_clear(t, 1000.0, 10000.0, 15000.0, None, S, kept)[1] == ''
    # a group starting near the record's start: its background falls below the clear bound 0 + 2P = 2000 yr
    group = rsp.onset_group(R_EP, 1000.0, [_ep(1, 1500, 9500), R_EP], _never)
    assert group.end_yr == 1500.0 and rsp.onset_clear(t, 1000.0, 10000.0, 15000.0, None, S, group)[1] == rsp.ONSET_EDGE
    # without a background the old stretch: [8000, 10000] is clear
    assert rsp.onset_clear(t, 1000.0, 10000.0, 15000.0, None, S)[1] == ''


def test_d_a_wave_of_an_earlier_episode_is_no_background():
    t = np.arange(0.0, 20001.0, 1.0)
    P = 500.0
    amp = np.where(t < 6000.0, 0.0005, np.where(t <= 14000.0, 0.005, 0.0))
    a = 3.0 + amp * np.sin(2 * np.pi * t / P)
    run, first = _ep(1, 9000, 14000), _ep(0, 6000, 8500)
    before, inside, _ = rsp.onset_bands(t, a, P, 9000.0, 14000.0)
    assert before / inside == pytest.approx(1.0, abs=0.05)  # the old background lies on the earlier episode's wave
    group = rsp.onset_group(run, P, [first, run], _never)
    assert (group.start_yr, group.end_yr, group.group_ids) == (5000.0, 6000.0, (0,))
    clear, why = rsp.onset_clear(t, P, 9000.0, 14000.0, None, S, group)
    before, inside_group, _ = rsp.onset_bands(t, a, P, 9000.0, 14000.0, clear=clear, before=(group.start_yr, group.end_yr))
    assert why == '' and inside_group == pytest.approx(inside)  # the episode itself is not changed
    assert 0.1 <= before / inside < S.mmr_o1_ratio_thin  # 0.0005 / 0.005 plus the band-pass leak of the onset


def test_d_gap_circulates_by_hand():
    """sigma_f flat at 0, the gap 300-700 yr: a rise of 1.1 turns between 400 and 600 yr is a circulation, 0.05 rad
    is not; a gap without finite samples counts as one; for 1:1 the orbit-averaged raw angle's range decides too."""
    t = np.arange(0.0, 1001.0, 1.0)
    left, right = _ep(0, 0, 300), _ep(1, 700, 1000)
    ramp = np.clip((t - 400.0) / 200.0, 0.0, 1.0)
    flat = SimpleNamespace(gap_raw_range=lambda left, right: math.nan)
    assert geo.gap_circulates(flat, t, 1.1 * 2 * np.pi * ramp, left, right)
    assert not geo.gap_circulates(flat, t, 0.05 * ramp, left, right)
    assert geo.gap_circulates(flat, t, np.where((t >= 300) & (t <= 700), np.nan, 0.0), left, right)
    raw = SimpleNamespace(gap_raw_range=lambda left, right: 7.0)  # > 2 pi
    assert geo.gap_circulates(raw, t, 0.05 * ramp, left, right)


def _bench(example):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    analysis = analyse_mmr(io.load_series(example), S)
    episode = next(p for p in analysis.episode_physics if p.episode_id == analysis.decisive_episode_id)
    return analysis, episode


def _gap_degrees(analysis, left_id, right_id):
    runs = {run.episode_id: run for run in analysis.geometry.runs}
    t, x = analysis.prep.t_yr, analysis.prep.angle_f
    mask = (t >= runs[left_id].t_end_yr) & (t <= runs[right_id].t_start_yr)
    return float(np.degrees(np.ptp(x[mask])))


@pytest.mark.benchmark
def test_d_va14_60kyr_measures_the_onset_of_the_group():
    analysis, episode = _bench('mmr3__2013_VA14__2J+6S-1+0+0-7__60kyr_1yr')
    verdict = analysis.verdict
    assert (str(verdict.status), str(verdict.angle_status), str(verdict.reason)) == ('uncertain', 'episodic', 'origin_unknown')
    assert analysis.decisive_episode_id == 5 and episode.onset_group_ids == (4,)
    assert _gap_degrees(analysis, 4, 5) < 360.0
    assert (episode.onset_before_t0_yr, episode.onset_before_t1_yr) == pytest.approx((36675.0, 40115.0), abs=1.0)
    ratio = episode.a_band_std_before_au / episode.a_band_std_in_au
    assert ratio == pytest.approx(0.73, abs=0.01)  # 1.028 before W21 D; the reviews' #4+#5: 0.73, 0.725
    assert ratio >= S.mmr_o1_ratio_fail  # still a fail: the row stays uncertain, never non_resonant
    # the explained column names the group: the wave O1 reads lies before #4, not on it
    row = {**analysis.numbers_row(), **analysis.summary_row()}
    text = ' '.join(vt.explain_mmr(analysis, S, row, analysis.episode_rows()).sentences)
    assert 'a strong a-wave precedes the episode (before/inside 0.733 ≥ 0.5, the wave before episode 4, which no' in text


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'example, earlier, expected, onset, short, ratio',
    [
        ('mmr3__17445__5J-2S-2+0+0-1', 2, ('transient', 'episodic', 'high'), False, rsp.ONSET_EPISODE, math.nan),
        ('mmr2__1863__7J-2+0-5', 0, ('transient', 'episodic', 'medium'), False, '', 0.456),
    ],
)
def test_d_an_earlier_episode_with_a_circulation_cuts_the_background(example, earlier, expected, onset, short, ratio):
    analysis, episode = _bench(example)
    verdict = analysis.verdict
    assert (str(verdict.status), str(verdict.angle_status), str(verdict.confidence)) == expected
    assert (str(Cap.ONSET_FAILED) in verdict.caps) is onset
    assert _gap_degrees(analysis, earlier, analysis.decisive_episode_id) >= 360.0
    assert episode.onset_group_ids == () and episode.onset_short == short
    earlier_end = next(run.t_end_yr for run in analysis.geometry.runs if run.episode_id == earlier)
    got = episode.a_band_std_before_au / episode.a_band_std_in_au
    if math.isnan(ratio):
        assert math.isnan(got)
    else:
        assert episode.onset_before_t0_yr == pytest.approx(earlier_end)
        assert got == pytest.approx(ratio, abs=0.005)


# --------------------------------------------------------------------------------------------- Q1d: the circulation exemption
# The author's labels (benchmark/author_label_decisions, W20 Q1d, 23.10) are the independent source: 1627 3V-4M-7 and
# 285637 3M+7J-6 (2 Myr) non_resonant/high, 1999 UQ 7E+2M-9 non_resonant/certain. By hand (confidence.apply_caps):
# raw high or certain with the cap short_period_aliased alone -> medium, without it -> the raw level. kinematic_drift
# keeps the cap; so does a circulation whose accepted run F6 removed as a forced wave (1036 9J-2S-3 at 5 yr: one wave,
# its identity |slope/c - 1| = 0.73 fails, and the identity reads a, which the folded lines spoil): raw certain -> medium,
# the cap binds, so the action is resample at its fold-free step (2 yr).

Q1D = [
    # example, (status, angle, reason, raw, confidence) with the exemption, its fold-free step
    ('mmr3__1627__3V-4M-7+0+0+8', ('non_resonant', 'circulation', 'circulation', 'high', 'high'), 0.25),
    ('mmr3__285637__3M+7J-6+0+0-4__2Myr', ('non_resonant', 'circulation', 'circulation', 'high', 'high'), 1.0),
    ('mmr3__1999_UQ__7E+2M-9+0+0+0', ('non_resonant', 'circulation', 'alias', 'certain', 'certain'), 0.5),
]


def _q1d(example, settings=S):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    return analyse_mmr(io.load_series(example), settings)


def _word(v):
    return (str(v.status), str(v.angle_status), str(v.reason), str(v.confidence_raw), str(v.confidence))


@pytest.mark.benchmark
@pytest.mark.parametrize('example, expected, step', Q1D)
def test_q1d_a_circulation_keeps_the_flag_not_the_cap(example, expected, step):
    result = _q1d(example)
    v = result.verdict
    assert _word(v) == expected
    assert CAP not in v.caps and str(Flag.SHORT_PERIOD_ALIASED) in result.sampling.flags
    assert (v.action, v.resample_step_yr) == (Action.NONE, None)
    assert result.sampling.fold_free_step_yr == step


@pytest.mark.benchmark
@pytest.mark.parametrize('example, expected, step', Q1D)
def test_q1d_the_switch_puts_the_cap_back(example, expected, step):
    result = _q1d(example, replace(S, mmr_short_period_cap_circulation=True))
    v = result.verdict
    assert _word(v)[:4] == expected[:4] and str(v.confidence) == 'medium' and list(v.caps) == [CAP]
    assert (v.action, v.resample_step_yr) == (Action.RESAMPLE, step)


@pytest.mark.benchmark
def test_q1d_a_forced_wave_circulation_keeps_the_cap():
    result = _q1d('mmr3__1036__9J-2S-3+0+0-4')
    v = result.verdict
    assert len(result.geometry.forced_waves) == 1
    assert _word(v) == ('non_resonant', 'episodic', 'circulation', 'certain', 'medium') and list(v.caps) == [CAP]
    assert (v.action, v.resample_step_yr) == (Action.RESAMPLE, 2.0)


@pytest.mark.benchmark
def test_q1d_a_kinematic_drift_keeps_the_cap():
    v = _q1d('mmr3__2002_BG__1E-10S-1+0+0+10').verdict
    assert (str(v.status), str(v.reason), str(v.confidence)) == ('non_resonant', 'kinematic_drift', 'medium')
    assert CAP in v.caps
