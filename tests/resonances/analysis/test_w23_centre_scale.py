"""W23 rule (a): the analysis at the scale of the centre (G2c) takes its turning points at the centre-turn prominence
p_c that qualified the scale (LIBRATION_CENTRE_SCALE_TP_AT_CENTRE), and the status-neutral diagnostics (d).

At W_c a turn must be a turn of the centre, as the precheck that qualified the scale reads it (the definition
`geometry.centre_turns_window_yr` uses): a swing below p_c is not resolved as one at that scale. Before W23 the
analysis at W_c re-selected its turning points at the 1-deg MMR floor, so the precheck (at p_c) and the analysis (at
1 deg) read different turning points.

The expected numbers are those of the W23 design runs (design/episodes: full-d is the W22 code, full-ad the
monkeypatched rule, changed-ad.csv the four rows that change), not the package's output:
- 2003 TX9 3E+2J-10 at 5 yr and 0.5 yr: one episode 2.21-17.49 kyr (11 TP, edge gaps 0.70/0.80 P) at W_c 1107.77 /
  1108.27 yr, uncertain (rule 5e: R0 measured and failed at that scale); without the rule 0.58-13.59 kyr (17 TP) /
  0.58-11.25 kyr (12 TP) at the fine scale;
- 523653 12N-19: 4.83-94.82 kyr with 9 TP instead of 11 (two 1-deg loops), resonant, high instead of medium;
- 385458 9N-17 300 kyr: 29.35-292.17 kyr (9 TP) at W_c 4608 yr instead of 68.12-264.79 kyr (7 TP) at 32216 yr.
The benchmark records are skipped without the benchmark data. The TX9 tests read the record with G2a off
(LIBRATION_ACTION_STEP, W24): G2a finds the Mars kick and cuts nothing (`test_tx9_g2a_finds_the_mars_kick_and_cuts_nothing`).
"""

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as geo
from resonances.resonance.analysis.mmr import analyse as analyse_module, decide, io
from resonances.resonance.analysis.mmr.models import MMRSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.settings import settings_from_config

S = settings_from_config()
OFF = replace(S, libration_centre_scale_tp_at_centre=False)
TX9 = 'mmr3__2003_TX9__3E+2J-10+0+0+5'
TX9_HALF = 'mmr3__2003_TX9__3E+2J-10+0+0+5__step0.5yr'
TX9_CUTOFF = {TX9: 1107.77, TX9_HALF: 1108.27}
S0 = replace(S, libration_action_step=False)  # TX9 before G2a (W24)


def _load(example):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    return io.load_series(example)


def _analysis(example, settings=S):
    return analyse_module.analyse_mmr(_load(example), settings)


def _accepted(analysis):
    """(t_start [kyr], t_end [kyr], n_tp) of every accepted episode."""
    return [(run.t_start_yr / 1e3, run.t_end_yr / 1e3, run.n_tp) for run in analysis.geometry.runs if run.accepted]


def test_the_setting_is_on_by_default():
    assert S.libration_centre_scale_tp_at_centre is True


@pytest.mark.benchmark
@pytest.mark.parametrize('example', [TX9, TX9_HALF])
def test_tx9_one_slow_libration_at_the_scale_of_the_centre(example):
    series = _load(example)
    analysis = analyse_module.analyse_mmr(series, S0)
    ((start, end, n_tp),) = _accepted(analysis)
    assert start == pytest.approx(2.21, abs=0.1) and end == pytest.approx(17.49, abs=0.1) and n_tp == 11
    (episode,) = [run for run in analysis.geometry.runs if run.accepted]
    gaps = (episode.t_start_yr - series.t_yr[0], series.t_yr[-1] - episode.t_end_yr)  # 0.70 and 0.80 P
    assert max(gaps) / episode.period_med_yr <= S.libration_coverage_periods
    assert analysis.prep.cutoff.binding == 'centre_scale'
    assert analysis.extra_columns['geo_centre_scale_cutoff_yr'] == pytest.approx(TX9_CUTOFF[example], abs=0.01)
    verdict = analysis.verdict
    assert (str(verdict.status), str(verdict.angle_status), str(verdict.angle_subtype)) == ('uncertain', 'libration', 'stationary')
    assert (str(verdict.reason), str(verdict.final_regime)) == ('origin_unknown', 'libration')
    assert analysis.extra_columns['rec_rule'] == '5e'  # R0 measured at W_c and failed (W23 correction 1)
    # (d): no union was rejected and no word rewritten
    assert analysis.extra_columns['geo_union_word_centre_scale'] == 0
    assert analysis.extra_columns['geo_cs_rej_status'] == '' and np.isnan(analysis.extra_columns['geo_cs_rej_cutoff_yr'])


@pytest.mark.benchmark
@pytest.mark.parametrize('example, fine', [(TX9, (0.58, 13.59, 17)), (TX9_HALF, (0.58, 11.25, 12))])
def test_tx9_without_the_rule_keeps_the_fine_episode(example, fine):
    analysis = _analysis(example, replace(OFF, libration_action_step=False))
    ((start, end, n_tp),) = _accepted(analysis)
    assert (round(start, 2), round(end, 2), n_tp) == fine
    assert analysis.prep.cutoff.binding != 'centre_scale'
    assert str(analysis.verdict.status) == 'uncertain'


@pytest.mark.benchmark
@pytest.mark.parametrize('settings, n_tp, confidence', [(S, 9, 'high'), (OFF, 11, 'medium')])
def test_523653_without_the_one_degree_loops(settings, n_tp, confidence):
    """12N-19 at W_c 3597 yr: two 1-deg loops were two more cycles and made L5 fail as support (medium)."""
    analysis = _analysis('mmr2__523653__12N-19+0+7', settings)
    ((start, end, n),) = _accepted(analysis)
    assert (round(start, 2), round(end, 2), n) == (4.83, 94.82, n_tp)
    assert analysis.extra_columns['geo_centre_scale_cutoff_yr'] == pytest.approx(3597.0)
    verdict = analysis.verdict
    assert (str(verdict.status), str(verdict.angle_status), str(verdict.confidence)) == ('resonant', 'libration', confidence)


@pytest.mark.benchmark
@pytest.mark.parametrize('settings, episode, cutoff', [(S, (29.35, 292.17, 9), 4608.0), (OFF, (68.12, 264.79, 7), 32216.0)])
def test_385458_first_centre_scale_suffices(settings, episode, cutoff):
    """9N-17 300 kyr: the first centre scale (3 P_cyc) succeeds without sub-p_c loops; the coarse retry is not needed."""
    analysis = _analysis('mmr2__385458__9N-17+0+8__300kyr', settings)
    ((start, end, n_tp),) = _accepted(analysis)
    assert (round(start, 2), round(end, 2), n_tp) == episode
    assert analysis.extra_columns['geo_centre_scale_cutoff_yr'] == pytest.approx(cutoff)
    assert (str(analysis.verdict.status), str(analysis.verdict.confidence)) == ('resonant', 'high')


@pytest.fixture
def centre_angle(monkeypatch):
    """The fine filtered angle that G2c reads, caught on its way into `_at_centre_scale`."""
    caught = {}
    original = analyse_module._at_centre_scale

    def catch(series, angle, *args):
        caught['angle'] = angle
        return original(series, angle, *args)

    monkeypatch.setattr(analyse_module, '_at_centre_scale', catch)
    return caught


@pytest.mark.benchmark
@pytest.mark.parametrize('example', [TX9, TX9_HALF])
def test_tx9_at_a_fixed_centre_scale_any_centre_prominence_finds_the_same_episode(example, centre_angle):
    """Sensitivity of rule (a) alone: at the W_c of the default run, every p_c in 8-18 deg qualifies the scale and the
    analysis there finds the precheck's episode (11 TP up to 13 deg, 9 from 16), 2.21-17.49 kyr. Below 8 deg the
    precheck's own run fails centre_moves, above 18 deg it no longer covers the record (scratchpad w23/impl/s4)."""
    series = _load(example)
    analysis = analyse_module.analyse_mmr(series, S0)
    coarse = geo.angle_at_centre_window(centre_angle['angle'], analysis.extra_columns['geo_centre_scale_cutoff_yr'], S0)
    for p_c, n_tp in ((8.0, 11), (10.0, 11), (13.0, 11), (16.0, 9), (18.0, 9)):
        precheck = decide.decisive_episode(geo.measure_geometry(coarse, math.radians(p_c), S0), S0)
        _, at_scale = analyse_module._decide_at_scale(series, coarse, math.radians(p_c), S0, None, False, False, '')
        ((start, end, n),) = _accepted(at_scale)
        assert (start, end) == pytest.approx((2.21, 17.49), abs=0.02) and n == n_tp == precheck.n_tp, p_c
        assert str(at_scale.verdict.status) == 'uncertain'


@pytest.mark.benchmark
@pytest.mark.parametrize('example, p_c', [(TX9, 13.0), (TX9_HALF, 7.0), (TX9_HALF, 13.0), (TX9_HALF, 16.0)])
def test_tx9_episode_over_the_centre_prominence(example, p_c):
    """The whole analysis with LIBRATION_CENTRE_TP_PROMINENCE_DEG moved from 10 deg: p_c sets W_c too (the turns of the
    centre, `geometry.centre_turns_window_yr`), so the scale moves with it; the episode stays. At 5 yr, p_c 7 and 16 deg
    give W_c 765 and 1555 yr whose precheck finds no covering episode, and the fine episode is reported (a fragility of
    the scale, not of rule a)."""
    analysis = _analysis(example, replace(S0, libration_centre_tp_prominence_deg=p_c))
    ((start, end, _),) = _accepted(analysis)
    assert (start, end) == pytest.approx((2.21, 17.49), abs=0.1)
    assert str(analysis.verdict.angle_status) == 'libration'


# W24 G2a (REPORT tx9 §3, the W24 review and its robustness fix, tx9-robust): half-amplitudes 144-146 deg before the
# transitional half-cycle 5.92-7.99 kyr, 12 deg after (separation 8.3, ratio 12), the swing of a /7.1: tests 1-6 pass.
# Test 7 (`geometry.regime_periods`) does not: V of the large swings (two wells, 132 and 274 deg at C = 0.1 x 2.35
# kyr) holds the first cycle after the step (7.99-10.24 kyr, law 0.98, its turning points within 0.14 of its half-
# swing), but the second (10.24-12.56 kyr) swings 0.74 of its half-swing off the loop of its energy and the third
# (12.56-16.48 kyr, across the 2.9-kyr gap of turning points) has the period law 1.61, at every cutoff 0.075-0.15. The
# late part is no constant-energy libration of that potential: not cut, one episode, uncertain as before W24.
TX9_STEP = {TX9: 6.96, TX9_HALF: 6.96}


@pytest.mark.benchmark
@pytest.mark.parametrize('example', [TX9, TX9_HALF])
def test_tx9_g2a_finds_the_mars_kick_but_one_potential_does_not_hold_the_late_cycles(example):
    analysis = _analysis(example)
    (step,) = analysis.geometry.action_steps
    assert not step.cuts and step.failed == geo.STEP_REGIME_PERIODS and step.time_yr / 1e3 == pytest.approx(TX9_STEP[example], abs=0.01)
    assert step.separation > 8.0 and step.ratio > 11.0 and step.a_ratio > 7.0 and step.dh < -0.5
    assert step.shape_failed == 'period_law@0.075' and step.shape_wells == 2 and step.shape_r2 > 0.9
    assert (step.shape_cycles_before, step.shape_cycles_after, step.shape_held_out) == (1, 3, False)
    assert step.law_before == pytest.approx(0.977, abs=0.01) and step.law_after == pytest.approx(1.61, abs=0.02)
    for item in step.shape_laws.split(';'):  # the worst cycle after the step fails at every cutoff
        assert float(item.split(':')[1].split('/')[1]) > 1 + S.libration_action_step_period_tol
    assert step.shape_tp_mismatch > 1.0 and step.shape_period_ratio == pytest.approx(0.627, abs=0.01)
    assert [(round(a, 2), round(b, 2)) for a, b, _ in _accepted(analysis)] == [(2.21, 17.49)]
    verdict = analysis.verdict
    assert (str(verdict.status), str(verdict.angle_status), str(verdict.reason)) == ('uncertain', 'libration', 'origin_unknown')
    assert 'action_step' not in verdict.flags and verdict.checks['check_mmr_potential'] == 'fail'


@pytest.mark.benchmark
@pytest.mark.parametrize('example', [TX9, TX9_HALF])
def test_tx9_r0_of_the_whole_record_is_guarded_like_every_check(example):
    """R0 (check_mmr_potential) is no longer set apart by the guard (W24 review): a cut that left it unmeasured would
    be withheld. TX9 is not cut, so the whole record's R0 fails in the reported analysis itself; its V_E at the
    record's cutoff (1108 yr) has one well, where the new law (each cycle in the well that holds it) is the old one."""
    assert not hasattr(analyse_module, 'POOLED_ACROSS_STEP')
    whole = _analysis(example, S0)
    assert whole.verdict.checks['check_mmr_potential'] == 'fail' and str(whole.verdict.status) == 'uncertain'
    assert [(round(a, 2), round(b, 2)) for a, b, _ in _accepted(whole)] == [(2.21, 17.49)]
    columns = _analysis(example).numbers_row()
    assert columns['geo_action_step_n_cuts'] == 0 and columns['geo_action_step_failed'] == 'regime_periods'


@pytest.mark.benchmark
@pytest.mark.xfail(strict=True, reason='W_c from the centre turns moves with p_c and the precheck fails there (not rule a)')
@pytest.mark.parametrize('p_c', [7.0, 16.0])
def test_tx9_five_years_keeps_the_episode_when_p_c_moves_the_scale(p_c):
    """The fragility the W23 S4 review measured: at 5 yr, p_c 7 / 16 deg give W_c 765 / 1555 yr, the precheck finds no
    covering episode there, G2c is not tried and the fine episode 0.58-13.59 kyr is reported. Passes once the choice of
    W_c is robust in p_c."""
    analysis = _analysis(TX9, replace(S, libration_centre_tp_prominence_deg=p_c))
    ((start, end, _),) = _accepted(analysis)
    assert (start, end) == pytest.approx((2.21, 17.49), abs=0.1)


# --- (d): the union guard's record of the analyses it rejects ---------------------------------------------------------


def _scripted(union_read: bool, tag: str):
    """A stand-in for a (coarse angle, analysis) pair of `_decide_at_scale`: only the column the guard reads."""
    return ('coarse', SimpleNamespace(extra_columns={'geo_union_read_episodic': float(union_read)}, tag=tag))


def test_the_union_guard_keeps_the_last_rejection(monkeypatch):
    monkeypatch.setattr(analyse_module, '_rejected_columns', lambda analysis: {'geo_cs_rej_status': analysis.tag})
    guard = analyse_module._union_guard
    assert guard(None, {}) == (None, {})  # no scale
    kept = _scripted(False, 'kept')
    assert guard(kept, {}) == (kept, {})  # kept, nothing rejected
    found, rejected = guard(_scripted(True, 'first'), {})
    assert found is None and rejected == {'geo_cs_rej_status': 'first'}
    assert guard(kept, rejected) == (kept, rejected)  # reject, then keep: the kept analysis carries the rejection
    assert guard(None, rejected) == (None, rejected)  # reject, then no scale
    assert guard(_scripted(True, 'second'), rejected) == (None, {'geo_cs_rej_status': 'second'})  # two rejects: the last


def test_a_record_stopped_before_the_geometry_has_empty_rejection_columns():
    """An invalid record (a <= 0) is stopped before the measurements; the geo_cs_rej_* columns are there, empty."""
    t = np.arange(0.0, 1000.0, 1.0)
    one = np.full_like(t, 0.1)
    series = MMRSeries(
        t_yr=t,
        a=np.full_like(t, -1.0),
        e=one,
        inc=one,
        Omega=one,
        omega=one,
        varpi=2 * one,
        angle_wrapped=one,
        resonance=ResonanceSpec.from_key('3N-4+0+1'),
        regime=Regime(step_yr=1.0, span_yr=999.0, n_rows=len(t)),
        name='invalid',
    )
    analysis = analyse_module.analyse_mmr(series, S)
    assert str(analysis.verdict.status) == 'invalid'
    columns = analysis.numbers_row()
    assert set(analyse_module.CS_REJECTED_NONE) <= set(columns)
    assert columns['geo_cs_rej_status'] == '' and np.isnan(columns['geo_cs_rej_cutoff_yr'])
