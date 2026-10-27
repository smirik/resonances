"""W16 integration: the jump locator's dominant step, rule 5g (a failed identity objects when L4 and R0 are not
measured), and the threshold profile of sigma0 in the subresonance scan's test L.

Expected numbers come from the constructions (a step placed at a known time, a libration of known half-amplitude), not
from the package.
"""

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.mmr import io
from resonances.resonance.analysis.mmr import subresonances as sr
from resonances.resonance.analysis.mmr.models import ResonanceSpec
from resonances.resonance.analysis.settings import AnalysisSettings
from tests.resonances.analysis.builders import make_geometry, make_turning_points

S = AnalysisSettings()
T = np.arange(0.0, 20000.5, 1.0)
TAU = 150.0  # 2 P_c at the physical cutoff of an ordinary MMR (P_c = 75 yr)
SWING_PERIOD = 700.0  # yr
SWING_AU = 0.01  # half-amplitude of a's own oscillation [au]
T_STEP = 12000.0


def _record(step_au):
    """a oscillates (period 700 yr, 0.01 au) to T_STEP, then rests at the mean + `step_au` (a circulation's flat a):
    the rest makes the robust spread of the increments small enough that the swings themselves pass the threshold, as
    in the W10 synthetic record."""
    a = np.where(T < T_STEP, SWING_AU * np.sin(2 * np.pi * T / SWING_PERIOD), step_au)
    return a


def test_a_step_well_above_the_swings_around_it_is_its_own_candidate():
    # the step is 3 x the swing's largest increment over tau (2 x 0.01 x sin(pi 150 / 700) = 0.0125 au)
    candidates = g.jump_candidates(T, _record(0.0375), TAU, SWING_PERIOD, S)
    assert len(candidates) == 1
    (candidate,) = candidates
    # the increment over tau of a step at T_STEP is largest within tau/2 of it
    assert abs(candidate.time_yr - T_STEP) <= TAU / 2
    assert candidate.duration_yr < SWING_PERIOD
    assert candidate.size == pytest.approx(0.0375, abs=SWING_AU)


def test_swings_without_a_step_stay_one_slow_group():
    # the same swings, a at rest at the mean afterwards: every run above the threshold is a swing of similar size, the
    # group spans 12 kyr and is no faster than one swing period: no candidate
    assert g.jump_candidates(T, _record(0.0), TAU, SWING_PERIOD, S) == []


@pytest.mark.parametrize('width_yr', [700.0, 1000.0, 1400.0])
def test_a_libration_whose_amplitude_grows_and_decays_has_no_candidate(width_yr):
    # Codex W16: sigma = A(t) sin(w t), A = 0.01 + 2.5 exp(-z^2) rad, z = (t - 30 kyr) / width, w = 2 pi / 700 yr;
    # a - a* = sigma_dot / c (c = 1 here: a unit choice). Bounded, smooth, no step of a: its largest swing stands out
    # above its neighbours by less than its own size
    t = np.arange(0.0, 60000.5, 1.0)
    w = 2 * np.pi / SWING_PERIOD
    z = (t - 30000.0) / width_yr
    amplitude = 0.01 + 2.5 * np.exp(-z * z)
    rate = 2.5 * np.exp(-z * z) * (-2 * z / width_yr)
    a = (rate * np.sin(w * t) + w * amplitude * np.cos(w * t)) * 1e-3  # au, any scale
    assert g.jump_candidates(t, a, TAU, SWING_PERIOD, S) == []


def test_increment_stretches_take_the_dominant_run_or_the_whole_group():
    increments = np.zeros(100)
    increments[10:15] = 1.0
    increments[20:25] = 1.2
    increments[30:35] = 5.0
    edges = np.array([10, 15, 20, 25, 30, 35])
    # threshold 0.5: 5.0 - 1.2 > max(0.5, 1.2), the third run alone
    assert g._increment_stretches(increments, edges, gap=10, threshold=0.5) == [(30, 35)]
    # a largest run of 2.0: 2.0 - 1.2 = 0.8 is above the threshold but below the neighbour's 1.2: one stretch
    increments[30:35] = 2.0
    assert g._increment_stretches(increments, edges, gap=10, threshold=0.5) == [(10, 35)]
    increments[30:35] = 5.0
    # threshold 4: no run dominates by more, one stretch from 10 to 35
    assert g._increment_stretches(increments, edges, gap=10, threshold=4.0) == [(10, 35)]
    # gaps of 5 samples are not below 5: three groups of one run
    assert g._increment_stretches(increments, edges, gap=5, threshold=0.5) == [(10, 15), (20, 25), (30, 35)]


# --- rule 5g on a benchmark record ---------------------------------------------------------------------------------


# Since W20 the strength of 2897's 35S-4 is measured in extended precision (10^36 x below what its period needs) and
# rule 5s decides first (test_w20_strength_mp.py); without that route rule 5g is what decides, and is tested here.
S_DOUBLE_ONLY = replace(S, mmr_strength_mp='off', mmr_kin_ripple_share_max=0.0)  # and without K2 (W20 CS7), which decides first


def _analysed(example, settings=S):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    return analyse_mmr(io.load_series(example), settings)


def test_2897_a_failed_identity_without_force_law_and_potential_is_no_dynamic_libration():
    analysis = _analysed('mmr2__2897__35S-4+0-31', S_DOUBLE_ONLY)
    verdict = analysis.verdict
    assert verdict.checks['check_mmr_a_response'] == 'not_measured'
    assert verdict.checks['check_mmr_potential'] != 'pass'
    assert verdict.checks['check_mmr_identity'] == 'fail'
    assert (str(verdict.status), str(verdict.reason), str(verdict.libration_origin)) == ('uncertain', 'origin_unknown', 'unknown')
    assert any(entry.name.startswith('5g') for entry in verdict.decision_path)


def test_48789_with_a_passing_identity_keeps_its_episode_dynamic():
    verdict = _analysed('mmr3__48789__3J-1S-1+0+0-1').verdict
    assert verdict.checks['check_mmr_identity'] == 'pass'
    # L4 and R0 not measured, the e^q scale is no measurement of the force: the geometry alone (rule 5u/6, W18)
    assert (str(verdict.status), str(verdict.libration_origin)) == ('transient', 'unknown')
    assert not any(entry.name.startswith('5g') for entry in verdict.decision_path)


def test_identity_objects_only_when_it_fails_on_an_episode_and_passes_on_none():
    from resonances.resonance.analysis.mmr.decide import identity_fails_locally
    from resonances.resonance.analysis.verdict import CheckValue

    def response(pooled, *local):
        return SimpleNamespace(identity=pooled, episode_responses=[SimpleNamespace(identity=x) for x in local])

    fail, ok, nm = CheckValue.FAIL, CheckValue.PASS, CheckValue.NOT_MEASURED
    # Codex W16: two pieces at different a* pass, their union fails: no objection
    assert not identity_fails_locally(response(fail, ok, ok))
    assert not identity_fails_locally(response(fail, fail, ok))
    assert not identity_fails_locally(response(fail))  # the pool alone
    assert not identity_fails_locally(response(fail, nm))
    assert identity_fails_locally(response(fail, fail, nm))  # 2897: the 11-TP episode fails, the 3-TP one unmeasured
    assert not identity_fails_locally(response(ok, fail))
    assert not identity_fails_locally(None)


# --- test L reads sigma0's threshold profile ------------------------------------------------------------------------


def test_member_turning_points_are_selected_at_the_stretch_thresholds_of_sigma0():
    # the member librates about pi with half-amplitude 15 deg (swings of 30 deg); sigma0's record threshold is 60 deg,
    # its stretch over the window 10 deg: at the record's alone no turning point survives, at the profile all do
    t = np.arange(20001, dtype=float)
    member = np.pi + math.radians(15.0) * np.sin(2 * np.pi * t / 500)
    window = (1000.0, 19000.0)

    def scan(local):
        turning_points = make_turning_points(prominence_rad=math.radians(60.0), local=local)
        return SimpleNamespace(
            t=t,
            window=window,
            settings=S,
            series=SimpleNamespace(resonance=ResonanceSpec.from_key('9M-16+0+7')),
            geometry=make_geometry(turning_points=turning_points),
            offset=lambda spec, m, nodal, smoothed=True: np.zeros_like(t),
            centre0=np.full_like(t, np.pi),
            screen_index=np.arange(len(t)),
        )

    prep = SimpleNamespace(angle_f=member, cutoff=SimpleNamespace(cutoff_yr=20.0))
    spec = ResonanceSpec.from_key('9M-16+2+5')
    assert sr.member_libration(scan(()), prep, spec, False).n_tp == 0
    local = (g.LocalThreshold(window[0], window[1], math.radians(10.0)),)
    # extrema of sin(2 pi t / 500) at 125 + 250 k inside [1000, 19000]: k = 4 .. 75, 72 turning points
    found = sr.member_libration(scan(local), prep, spec, False)
    assert found.n_tp == 72 and found.librates


def test_2897_figure_text_names_the_failed_identity_with_the_record_numbers():
    from resonances.plotting import verdict_text as vt

    analysis = _analysed('mmr2__2897__35S-4+0-31', S_DOUBLE_ONLY)
    row = {**analysis.summary_row(), **analysis.numbers_row()}
    facts = vt.MMRFacts(analysis, S_DOUBLE_ONLY, row, analysis.episode_rows())
    step = vt._force_step(facts)
    assert facts.rule == '5g' and step.state == vt.StepState.LEAF
    response = analysis.response
    assert f'identity slope/c {vt.fmt(response.identity_slope_ratio)}' in step.outcome
    assert f'corr {vt.fmt(response.identity_corr)}' in step.outcome
    assert response.identity_slope_ratio < 0  # sigma_dot runs against a: no pendulum of this angle
    assert step.outcome.endswith('→ uncertain')
