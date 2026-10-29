"""W20 CS2: the two-body strength by the resonant harmonic in extended precision, rule 5s before 5a*/5e/5g/5u, and
the display columns of the strength (R1 §1).

Expected values come from sources independent of the code under test:
- f27(alpha) of the interior 2:1 from Murray & Dermott 1999, Table 8.5 (1.19049 at alpha = 2^(-2/3)), and the same
  number from the Laplace coefficient b_{1/2}^{(2)} integrated here with scipy (M&D eq. 6.67 and Table B.4:
  f27 = (1/2)(-4 - alpha D) b_{1/2}^{(2)});
- the double-precision profile `averaged_profile` where it resolves the harmonic (a different quadrature);
- the d'Alembert law: a coplanar term of order q with a circular planet is proportional to e^q;
- the benchmark rows: lower bounds from the W20 B prototype (an independent script, scratchpad answers-1019/B) and
  385266's T0 range from the R1 prototype.
"""

import math
import resource
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad

from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import averaged_potential as av
from resonances.resonance.analysis.mmr import decide, io
from resonances.resonance.analysis.mmr import strength as st
from resonances.resonance.analysis.mmr.coorbital import Orbit
from resonances.resonance.analysis.mmr.decide import AngleDecision
from resonances.resonance.analysis.mmr.models import Strength
from resonances.resonance.analysis.settings import AnalysisSettings, validate_settings
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, CheckValue, Reason, Status
from tests.resonances.analysis.builders import make_response

S = AnalysisSettings()
S_NO_K2 = replace(S, mmr_kin_ripple_share_max=0.0)  # no episode swing is below 0 x the centre range: K2 never fires
LIMIT = S.mmr_strength_implausible_ratio
A_JUPITER = 5.2026
ALPHA_21 = 2 ** (-2 / 3)  # exact 2:1 commensurability
F27_TABLE = 1.19049  # |f27| at alpha = 0.6300, Murray & Dermott 1999, Table 8.5
EPISODIC = AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, 'G2', 0)


def _rule(**changes):
    return replace(st.harmonic_rule(S), **changes)


def _ellipse(a, e, inc=0.0, omega=0.0, node=0.0):
    return av.Ellipse(a, Orbit(e, inc, omega, node))


CIRCULAR_JUPITER = _ellipse(A_JUPITER, 0.0)


# --- 1. normalisation against M&D and the Laplace coefficient ----------------------------------------------------


def _laplace(j: int, alpha: float) -> float:
    """b_{1/2}^{(j)}(alpha) = (1/pi) int_0^{2 pi} cos(j psi) / sqrt(1 - 2 alpha cos psi + alpha^2) dpsi."""
    value, _ = quad(lambda psi: np.cos(j * psi) / np.sqrt(1 - 2 * alpha * np.cos(psi) + alpha**2), 0, 2 * np.pi, epsabs=1e-14, epsrel=1e-13)
    return value / np.pi


def test_laplace_coefficient_reproduces_the_table_value():
    step = 1e-4
    derivative = (_laplace(2, ALPHA_21 + step) - _laplace(2, ALPHA_21 - step)) / (2 * step)
    f27 = 0.5 * (-4 * _laplace(2, ALPHA_21) - ALPHA_21 * derivative)
    assert abs(abs(f27) - F27_TABLE) < 1e-5


@pytest.mark.parametrize('e', [0.005, 0.01, 0.02])
def test_interior_two_to_one_harmonic_is_f27_e_over_a_jupiter(e):
    # sigma = 2 lambda_J - lambda - varpi: k = -1, k_P = 2; <D> = (f27 e / a_J) cos sigma, so 2|C| a_J / e = |f27|
    body = _ellipse(ALPHA_21 * A_JUPITER, e, omega=0.3)
    harmonic = av.harmonic_amplitude_mp(body, CIRCULAR_JUPITER, -1, 2, 0.3, _rule(nodes_body=64, nodes_planet=32))
    assert harmonic.measured and harmonic.converged
    assert abs(harmonic.curvature * A_JUPITER / e - F27_TABLE) < 1e-4


# --- 2. agreement with double precision where it resolves -----------------------------------------------------------


def test_seven_to_two_matches_the_first_harmonic_of_the_double_profile():
    # 7:2 with an eccentric inclined Jupiter, e = 0.1: the harmonic is ~1e-6 of D0, double precision resolves it
    body = _ellipse(A_JUPITER * (2 / 7) ** (2 / 3), 0.1, inc=0.02, omega=0.4, node=0.2)
    planet = _ellipse(A_JUPITER, 0.048, inc=0.0228, omega=0.25, node=1.75)
    phase = 5 * (0.4 + 0.2)  # sigma = 2 lambda_J - 7 lambda + 5 varpi
    harmonic = av.harmonic_amplitude_mp(body, planet, -7, 2, phase, _rule())
    sigma, values = av.averaged_profile(body, planet, -7, 2, phase, 720)
    first = np.fft.rfft(values)[1] / len(values)  # <D> = D0 + 2|c1| cos(sigma + arg c1) + ...
    assert abs(harmonic.curvature / (2 * abs(first)) - 1) < 1e-6
    assert abs(harmonic.mean / np.mean(values) - 1) < 1e-9
    # its minimum is the profile's first-harmonic minimum, pi - arg c1
    assert float(av.circular_distance(harmonic.centre_rad, np.pi - np.angle(first))) < 1e-6


# --- 3. the e^q law --------------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_coplanar_order_eleven_term_scales_as_e_to_the_eleventh():
    # 23:12 with a circular planet: sigma = 12 lambda_J - 23 lambda + 11 varpi, amplitude ~ e^11 (+ O(e^13))
    a = A_JUPITER * (12 / 23) ** (2 / 3)
    values = [av.harmonic_amplitude_mp(_ellipse(a, e, omega=0.3), CIRCULAR_JUPITER, -23, 12, 0.0, _rule()) for e in (0.01, 0.02)]
    assert all(v.measured for v in values)
    assert abs(math.log2(values[1].value / values[0].value) - 11) < 0.1


# --- 4. convergence and the precision doubling ------------------------------------------------------------------------


def _2897_like():
    # 35S-4 at e = 0.09 (like 2897): the harmonic is ~1e-30 of D0, far below double precision
    body = _ellipse(9.5371 * (4 / 35) ** (2 / 3), 0.0927, inc=0.0819, omega=1.0, node=0.3)
    planet = _ellipse(9.5371, 0.054, inc=0.043, omega=1.5, node=1.98)
    return body, planet


@pytest.mark.slow
def test_doubling_the_nodes_changes_the_harmonic_by_less_than_the_tolerance():
    body, planet = _2897_like()
    rule = _rule()
    base = av.harmonic_amplitude_mp(body, planet, -4, 35, 0.0, rule)
    fine = av.harmonic_amplitude_mp(body, planet, -4, 35, 0.0, _rule(nodes_body=2 * rule.nodes_body, nodes_planet=2 * rule.nodes_planet))
    assert base.measured and base.dps == S.mmr_strength_mp_dps
    assert base.value / abs(base.mean) < 1e-20  # far below what double precision resolves (1e-12)
    assert abs(fine.value / base.value - 1) < S.mmr_strength_quadrature_tol


def test_too_few_planet_nodes_is_reported_not_used():
    # N_L = 32 aliases k_P - N_L = 3 onto a term of order 1: grid and twin disagree
    body, planet = _2897_like()
    harmonic = av.harmonic_amplitude_mp(body, planet, -4, 35, 0.0, _rule(nodes_planet=32))
    assert harmonic.reason == av.REASON_NOT_CONVERGED and not harmonic.measured


def test_a_mixed_alias_is_seen_by_the_one_axis_twins():
    """FIX1 (Codex A #1): with N_M = N_L = 8 the 2:1 (k = -1, k_P = 2) aliases onto (2 - 8, -1 + 8) = (-6, 7), a term of
    order 1 (r = -1, s = +1): a twin shifted by half a node in both angles turns it by (-1)^(r+s) = 1 and saw nothing
    (the value 2.1 % off M&D's f27 e / a_J was reported converged). Each one-axis twin turns it by -1: not converged.
    With N_M = N_L = 32 the alias is far: converged and within 1e-4 of the table (its 6 digits and e^3 terms)."""
    e = 0.01
    expected = F27_TABLE * e / (2 * A_JUPITER)  # |C| = |f27| e / (2 a_J)
    body = _ellipse(ALPHA_21 * A_JUPITER, e, omega=0.3)
    coarse = av.harmonic_amplitude_mp(body, CIRCULAR_JUPITER, -1, 2, 0.3, _rule(nodes_body=8, nodes_planet=8, dps=30))
    assert coarse.reason == av.REASON_NOT_CONVERGED and not coarse.converged
    fine = av.harmonic_amplitude_mp(body, CIRCULAR_JUPITER, -1, 2, 0.3, _rule(nodes_body=32, nodes_planet=32, dps=30))
    assert fine.measured and abs(fine.value / expected - 1) < 1e-4


@pytest.mark.slow
def test_digits_are_doubled_until_the_harmonic_is_resolved_or_reported():
    body, planet = _2897_like()
    # 20 digits with a 15-digit guard cannot hold a 1e-30 harmonic: doubled to 40, then 80 > max 60 -> 60
    harmonic = av.harmonic_amplitude_mp(body, planet, -4, 35, 0.0, _rule(dps=20, dps_max=60, nodes_body=128, nodes_planet=128))
    assert harmonic.measured and harmonic.dps == 60
    exhausted = av.harmonic_amplitude_mp(body, planet, -4, 35, 0.0, _rule(dps=20, dps_max=40, nodes_body=128, nodes_planet=128))
    assert exhausted.reason == av.REASON_PRECISION and exhausted.dps == 40


def test_mp_settings_validate():
    with pytest.raises(ValueError):
        validate_settings(replace(S, mmr_strength_mp='always'))
    with pytest.raises(ValueError):
        validate_settings(replace(S, mmr_strength_mp_guard_digits=60))  # the guard must stay below the digits


def test_double_floor_routes_a_rounding_curvature_to_the_harmonic():
    rounding = av.CentreCurvature(1e-15, 1e-15, 0.0, 0.0, 1.0, 720, '', 1.0, level=0.1)  # 1e-14 of D0
    resolved = replace(rounding, curvature=1e-9, curvature_half=1e-9)
    assert st.needs_harmonic(rounding, S) and not st.needs_harmonic(resolved, S)
    assert st.needs_harmonic(replace(resolved, reason=av.REASON_NO_CENTRE), S)
    assert not st.needs_harmonic(rounding, replace(S, mmr_strength_mp='off'))


# --- 5. rule 5s before 5g/5a* ------------------------------------------------------------------------------------------


def _response(l4=CheckValue.NOT_MEASURED):
    """A response whose identity fails pooled and on its one episode (old leaf 5g; 2897: slope -2.86 c)."""
    return make_response(
        a_response=l4,
        force_r2_h1=0.1,
        force_r2_h2=0.1,
        force_law='single',
        multiplet=None,
        coorbital=None,
        identity=CheckValue.FAIL,
        identity_slope_ratio=-2.86,
        identity_corr=-0.9,
        episode_responses=[SimpleNamespace(identity=CheckValue.FAIL)],
    )


def _strength(ratio, method=st.METHOD_HARMONIC, reason='', three_body=False, residual=float('nan')):
    return Strength(
        mu_scale=1e-12,
        mu_scale_no_eq=1e-4,
        eq_legal=True,
        mu_eff_required=1e-6,
        ratio=ratio,
        omega0_theory_rad_yr=float('nan'),
        period_theory_yr=float('nan'),
        implausible=ratio >= LIMIT,
        check=CheckValue.FAIL if ratio > S.mmr_strength_check_ratio else CheckValue.PASS,
        method=method,
        not_measured_reason=reason,
        three_body=three_body,
        divisor_residual=residual,
    )


def _decide(strength, response=None, settings=S):
    ledger = Ledger.from_settings(settings)
    response = _response() if response is None else response
    return decide._force_and_potential(EPISODIC, None, response, None, settings, ledger, strength), ledger


def test_implausible_harmonic_strength_decides_before_the_failed_identity():
    decision, ledger = _decide(_strength(1e6))
    assert (decision.status, decision.reason, decision.rule) == (Status.NON_RESONANT, Reason.IMPLAUSIBLE_STRENGTH, '5s')
    entry = next(e for e in ledger if e.name.startswith(decide.STRENGTH_FIRST))
    assert entry.value == 1e6 and entry.threshold == LIMIT
    assert not any(e.name.startswith('5g') for e in ledger)


def test_direct_averaging_too_and_a_failed_force_law_does_not_shield_it():
    decision, _ = _decide(_strength(1e6, st.METHOD_DIRECT), _response(CheckValue.FAIL))  # old leaf 5a*
    assert decision.rule == '5s' and decision.status == Status.NON_RESONANT


def test_centre_outside_well_keeps_the_old_order_unless_the_setting_says_so():
    strength = _strength(1e6, st.METHOD_DIRECT, st.REASON_CENTRE_OUTSIDE_WELL)
    decision, ledger = _decide(strength)
    assert decision.rule == '5g' and decision.status == Status.UNCERTAIN
    assert not any(e.name.startswith(decide.STRENGTH_FIRST) for e in ledger)
    flipped = replace(S, mmr_strength_first_outside_well=True)
    assert _decide(strength, settings=flipped)[0].rule == '5s'


def test_a_plausible_ratio_leaves_the_identity_in_charge():
    decision, _ = _decide(_strength(50.0))
    assert decision.rule == '5g' and decision.status == Status.UNCERTAIN


def test_three_body_only_when_the_small_divisors_cannot_close_the_gap():
    # SR158 1V-4S-5: ratio 8.9e9, residual 5.1e6 after B = 42 -> 5s; 46596: residual 22 -> the old leaf, never 5s*
    closed = _strength(8.9e9, st.METHOD_SCALE, three_body=True, residual=5.1e6)
    assert _decide(closed)[0].rule == '5s'
    open_gap = _strength(4.1e7, st.METHOD_SCALE, three_body=True, residual=22.3)
    assert _decide(open_gap)[0].rule == '5g'


def test_harmonic_strength_counts_as_a_physics_measurement():
    assert 'F' in st.physics_margins(('L4', 'F'), _strength(1.0))
    assert 'F' not in st.physics_margins(('L4', 'F'), _strength(1.0, st.METHOD_SCALE))


# --- 6. benchmark rows (skip without data) -----------------------------------------------------------------------------


def _analysed(example, settings=S):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    series = io.load_series(example)
    return series, analyse_mmr(series, settings)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'example, lower',
    [
        ('mmr2__2001_PG14__33S-2+0-31', 1e20),  # prototype 6.8e23
        ('mmr2__2897__35S-4+0-31', 1e30),  # prototype 2.3e37
        ('mmr2__16126__2R-35+0+33', 1e18),  # prototype 8.2e20
    ],
)
def test_high_order_rows_need_an_implausible_strength(example, lower):
    # 2897: since W20 CS7 the kinematic drift of its slow centre (K2, step 3) decides before rule 5s; with K2 off
    # (no ripple share passes) rule 5s is what decides, and is tested here
    series, analysis = _analysed(example, S_NO_K2)
    strength, verdict = analysis.strength, analysis.verdict
    assert strength.method == st.METHOD_HARMONIC and strength.ratio >= lower
    assert str(verdict.status) == 'non_resonant' and str(verdict.reason) == 'implausible_strength'
    assert any(e.name.startswith(decide.STRENGTH_FIRST) for e in verdict.decision_path)
    # the extended-precision route alone, timed and its memory bounded
    episode = next(e for e in analysis.geometry.runs if e.episode_id == analysis.decisive_episode_id)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.process_time()  # CPU time of this process: the wall clock failed under load (5.1 and 5.8 s, W23)
    theory = st.direct_theory(series, analysis.context, episode, analysis.prep.cutoff.cutoff_yr, S)
    assert theory.method == st.METHOD_HARMONIC and theory.measured
    assert time.process_time() - start <= 10.0
    grown = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss  # bytes on macOS, KiB on Linux
    assert grown <= 200 * 1024 * 1024


@pytest.mark.benchmark
def test_sr158_three_body_is_decided_by_5s_before_its_failed_identity():
    _, analysis = _analysed('mmr3__2011_SR158__1V-4S-5+0+0+8')
    verdict = analysis.verdict
    assert (str(verdict.status), str(verdict.reason)) == ('non_resonant', 'implausible_strength')
    entry = next(e for e in verdict.decision_path if e.name.startswith(f'{decide.STRENGTH_FIRST} small divisors'))
    assert entry.value >= 1e6  # residual 5.1e6 (W17 Q3)


@pytest.mark.benchmark
def test_the_figure_text_names_the_harmonic_strength():
    from resonances.plotting import verdict_text as vt

    _, analysis = _analysed('mmr2__2897__35S-4+0-31', S_NO_K2)
    row = {**analysis.summary_row(), **analysis.numbers_row()}
    facts = vt.MMRFacts(analysis, S, row, analysis.episode_rows())
    step = vt._force_step(facts)
    # L4 and R0 are both not measured on 2897: the branch's wording, with the harmonic's ratio
    assert facts.rule == '5s' and not vt.strength_first(facts) and step.state == vt.StepState.LEAF
    assert step.outcome.startswith('both not measured') and 'resonant harmonic' in step.outcome
    assert step.outcome.endswith('→ non_resonant')


def test_strength_first_wording_only_when_a_check_was_measured():
    from resonances.plotting import verdict_text as vt

    path = {f'{decide.STRENGTH_FIRST} required / directly averaged strength (two-body)': None}
    failed = SimpleNamespace(rule='5s', checks={'a_response': 'fail', 'potential': 'not_measured'}, path=path)
    unmeasured = SimpleNamespace(rule='5s', checks={'a_response': 'not_measured', 'potential': 'not_measured'}, path=path)
    assert vt.strength_first(failed) and not vt.strength_first(unmeasured)
    assert vt.STRENGTH_FIRST == decide.STRENGTH_FIRST


# --- 7. the R1 display columns -----------------------------------------------------------------------------------------


@pytest.mark.benchmark
def test_385266_strength_columns_give_the_theoretical_period_range():
    _, analysis = _analysed('mmr2__385266__16N-27+0+11')
    columns = analysis.strength.to_columns()
    # R1 prototype: T0_th 209-260 kyr over the record, 209-234 kyr on 55-95 kyr
    assert 200 <= columns['str_P_lib_theory_min_kyr'] <= columns['str_P_lib_theory_max_kyr'] <= 270
    assert columns['str_T_obs_source'].startswith(('episode:', 'rejected_run:'))
    assert columns['str_snap_t0_yr'] < columns['str_snap_t1_yr']
    # T_obs back from mu_required: the same formula as the theory's period
    k, n = 27, analysis.context.mean_motion_rad_yr
    expected = 2 * np.pi / (k * n * np.sqrt(3 * columns['str_mu_eff_req']))
    assert columns['str_T_obs_yr'] == pytest.approx(expected, rel=1e-12)
    # the separatrix half-width 2 omega0 / |c| = (4/3) a sqrt(3 mu): 0.0023-0.0028 au in the prototype
    assert 0.002 <= columns['str_sep_halfwidth_min_au'] <= columns['str_sep_halfwidth_max_au'] <= 0.0032
    assert 0 < columns['str_e_snap_min'] <= columns['str_e_snap_max'] < 1
    assert columns['str_ratio_all_min'] <= columns['str_ratio_min']
