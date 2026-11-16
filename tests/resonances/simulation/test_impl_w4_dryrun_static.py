"""Static dry-run (SPEC §9): elements only, no integration.

Expected numbers come from outside the module: hand formulas written out here (ghost axis, crossing bands,
tau_peri, Laplace coefficient by scipy quadrature), the research prototype of 2026-10-15 (SAMPLING.md §2, §3.2, §11:
Laplace-Lagrange 51.3 arcsec/yr for 21514, ghost 1.0941756 au for 1999 UQ at 5.0012 yr) and the author's 1-yr rerun
of 1999 UQ (AUTHOR-FEEDBACK item 10: 3999 turns in 20 kyr, i.e. 0.2 turns per 1-yr step).
"""

import json
import math

import pytest
from scipy import integrate

import resonances
from resonances.simulation import dry_run as dr
from resonances.simulation.config import nout_for_step

DEG = math.pi / 180
UQ = dict(a=1.0940602, e=0.0155969, inc=0.4331147985225309, Omega=3.687738465129981, omega=0.13104333449701386, M=6.077459508522661)
CERBERUS = dict(a=1.079942, e=0.466887, inc=16.10174 * DEG, Omega=212.87978 * DEG, omega=325.29655 * DEG, M=220.49748 * DEG)
HILDA = dict(
    a=3.9683768284310585,
    e=0.1384847450994033,
    inc=0.1366838033702228,
    Omega=3.980744967033935,
    omega=0.6818375012739626,
    M=2.4215746014696986,
)
N_EARTH, N_MARS = 360.000505, 191.406585  # deg / Gaussian yr, const.PLANET_N_DEG_PER_GAUSS_YR


def simulation(**kwargs):
    defaults = dict(name='dryrun-test', date='2026-09-14', save=None, plot=None, save_summary=False)
    return resonances.Simulation(**{**defaults, **kwargs})


def nea_run(nout):
    sim = simulation(integration_years=20000, Nout=nout, integrator='ias15')
    sim.add_body(UQ, ['7E+2M-9+0+0+0', '8M-9J-4+0+0+5'], name='1999 UQ')
    return sim


def test_1999_uq_aliased_at_the_production_step_and_resolved_at_one_year():
    step = int(20000 * 2 * math.pi) / (2 * math.pi) / 3999  # what Nout = 4000 gives: 5.0012 yr
    report = nea_run(4000).dry_run('static')
    row = report.rows.set_index('resonance').loc['7E+2M-9+0+0+0']
    assert row.step_yr == pytest.approx(5.0012, abs=1e-4)
    assert row.verdict == 'aliased_unambiguous' and row.branches == '1'
    assert row.expected_angle_status == 'circulation' and row.action == 'predictive_unwrap'
    # the ghost of branch 1 by hand: 7 n_E + 2 n_M - 9 n(a_1) = 360 deg / dt (no varpi terms in this angle)
    n_ghost = (7 * N_EARTH + 2 * N_MARS - 360.0 / step) / 9
    ghost = (360.0 / n_ghost) ** (2 / 3)
    assert ghost == pytest.approx(1.0941756, abs=2e-7)  # SAMPLING.md §3.2
    assert row.ghost_axis_au == pytest.approx(ghost, rel=1e-9)
    assert row.nearest_ghost_au == pytest.approx(1.2e-4, abs=1e-5)  # "1.2e-4 au: would have been caught"
    assert row.nyquist_halfwidth_au == pytest.approx(UQ['a'] ** 2.5 / (3 * 9 * step), rel=1e-9)
    assert any('7E+2M-9+0+0+0' in w and 'aliased_unambiguous' in w for w in report.warnings)
    # the slow three-body angle is resolved at the same step
    assert report.rows.set_index('resonance').loc['8M-9J-4+0+0+5'].verdict == 'resolved'

    one_year = nea_run(20001).dry_run('static').rows.set_index('resonance').loc['7E+2M-9+0+0+0']
    assert one_year.verdict == 'resolved' and one_year.branches == '0'
    assert one_year.turns_per_step == pytest.approx(0.2, abs=0.002)  # 3999 turns / 20000 steps


def test_integrator_recommendation_for_an_nea_a_crosser_and_a_hilda():
    sim = simulation(integration_years=20000)  # SABA(10,6,4), dt = 1 (58 d): wrong for both NEAs
    sim.add_body(UQ, ['7E+2M-9+0+0+0'], name='1999 UQ')
    sim.add_body(CERBERUS, ['10E-11+0+1'], name='1865')
    sim.add_body(HILDA, ['3J-2+0-1'], name='153')
    report = sim.dry_run('static')
    bodies = report.bodies.set_index('body')
    assert bodies.loc['1999 UQ', 'class'] == 'nea' and bodies.loc['1999 UQ', 'integrator_recommended'] == 'ias15'
    assert bodies.loc['1999 UQ', 'crossing'] == ''  # Q_E + R_H = 1.0267 < q = 1.077
    # Cerberus q = 0.576, Q = 1.584: Venus (Q 0.728), Earth, Mars (q 1.381); not Mercury (Q 0.467)
    assert bodies.loc['1865', 'crossing'] == 'Venus,Earth,Mars' and bodies.loc['1865', 'integrator_recommended'] == 'ias15'
    hilda = bodies.loc['153']
    assert hilda['class'] == 'main' and hilda['integrator_recommended'] == 'SABA(10,6,4)'
    tau = hilda['P_yr'] * math.sqrt((1 - HILDA['e']) ** 3 / (1 + HILDA['e']))
    assert hilda['tau_peri_yr'] == pytest.approx(tau)
    assert hilda['dt_recommended_yr'] == pytest.approx(1 / (2 * math.pi))  # 58 d < P/10 = 0.79 yr, tau/10 = 0.59 yr
    assert sum('1999 UQ' in w and 'ias15' in w for w in report.warnings) == 1
    assert report.run['integrator_recommended'] == 'ias15' and report.run['mixed_integrators']
    changes = report.apply(step=False, integrator=True)
    assert changes == {'integrator': 'ias15'} and sim.config.integrator == 'ias15'


def test_hilda_cutoff_and_the_step_limits():
    sim = simulation(integration_years=100000)  # default Nout: 15.92 yr
    sim.add_body(HILDA, ['3J-2+0-1'], name='153')
    report = sim.dry_run('static')
    row = report.rows.iloc[0]
    step = sim.config.output_step_yr
    assert step == pytest.approx(15.918, abs=1e-3)
    # P_c = max(1.5 x P_syn,J = 35.6 yr, 150 yr / 2 = 75 yr): the floor bound, whatever the step
    assert row.cutoff_pred_yr == pytest.approx(75.0) and row.cutoff_binding == 'floor'
    assert row.step_valid_yr == pytest.approx(37.5) and row.expected_status == ''  # 15.9 < 75 / 2: filterable
    # short periods: P = 3.96838^1.5 = 7.9053 yr, P_syn,J = 1 / (1/7.9053 - 30.349646/360) = 23.70 yr: min / 2 = 3.953 yr
    period = HILDA['a'] ** 1.5
    assert row.step_short_period_yr == pytest.approx(period / 2, rel=1e-9)
    assert row.expected_flags == 'short_period_aliased' and 'short_period_aliased' in row.warnings
    # limits: short period 3.95 yr (-> 2.5 yr), libration 150 / 20 = 7.5 yr
    assert report.run['step_limits_yr']['libration'] == pytest.approx(7.5)
    assert report.run['step_auto_yr'] == 2.5 and report.run['binding_rule'] == 'short_period'
    assert report.run['Nout_auto'] == 40001
    assert row.verdict == 'resolved' and row.action == 'none'
    assert any('the short_period limit 3.953 yr' in w for w in report.warnings)


TROJAN = dict(a=5.2026, e=0.05, inc=0.18, Omega=1.0, omega=1.5, M=0.3)  # Jupiter's a: a tadpole about L4/L5
TADPOLE_YR = 147.75  # P_J / sqrt(27 mu_J / 4), P_J = 11.8617 yr, m_Sun / m_J = 1047.3486


@pytest.mark.parametrize('years, nout, undersampled', [(100000, 6283, False), (100000, 5001, True)])
def test_trojan_step_validity_and_its_resample_step(years, nout, undersampled):
    # P_floor = 0.5 x 147.75 = 73.9 yr, P_c = 36.9 yr: dt = 15.92 yr filters, dt = 20 yr (>= P_c / 2 = 18.47 yr) does
    # not: invalid / undersampled, resample at the largest round step <= 73.9 / 20 = 3.69 yr: 2.5 yr
    sim = simulation(integration_years=years, Nout=nout)
    sim.add_body(TROJAN, ['1J-1+0+0'], name='trojan')
    report = sim.dry_run('static')
    row = report.rows.iloc[0]
    assert row.libration_floor_yr == pytest.approx(TADPOLE_YR / 2, rel=2e-3)
    assert row.cutoff_pred_yr == pytest.approx(TADPOLE_YR / 4, rel=2e-3) and row.cutoff_binding == 'floor'
    assert row.step_valid_yr == pytest.approx(TADPOLE_YR / 8, rel=2e-3)
    assert row.step_short_period_yr == pytest.approx(TROJAN['a'] ** 1.5 / 2, rel=1e-6)  # 5.93 yr: P_syn,J is huge
    assert row.expected_flags == 'short_period_aliased'
    if undersampled:
        assert row.step_yr == pytest.approx(20.0, rel=1e-3)
        assert (row.expected_status, row.expected_reason, row.action) == ('invalid', 'undersampled', 'resample')
        assert row.resample_step_yr == 2.5 and 'undersampled (resample at 2.5 yr)' in row.warnings
    else:
        assert row.step_yr == pytest.approx(15.92, abs=0.01)
        assert row.expected_status == '' and math.isnan(row.resample_step_yr)
    # the auto step: libration 73.9 / 20 = 3.69 yr binds before the short periods (5.93 yr) and the filter (18.47 yr)
    assert report.run['step_limits_yr']['filter'] == pytest.approx(TADPOLE_YR / 8, rel=2e-3)
    assert report.run['step_auto_yr'] == 2.5 and report.run['binding_rule'] == 'libration'


def test_auto_step_avoids_the_bounds_it_would_be_flagged_at():
    # a short-period bound that is itself a round step: P_orb / 2 = 1 yr at a = 2^(2/3) au (P = 2 yr) and no planet of
    # the angle faster. dt = 1 yr would be flagged (>=), so the auto step is the next round value below, 0.5 yr
    assert dr.snap_step(math.nextafter(1.0, 0.0), (0.5, 1.0, 2.0)) == 0.5
    assert dr.snap_step(1.0, (0.5, 1.0, 2.0)) == 1.0


def test_trojan_with_an_explicit_cutoff_gets_a_step_that_filters_it():
    # MMR_FILTER_CUTOFF_YR = 1 yr: P_c / 2 = 0.5 yr bounds the resample step (0.25 yr) and the auto step ('filter')
    sim = simulation(integration_years=100000, Nout=6283, mmr_filter_cutoff_yr=1.0)
    sim.add_body(TROJAN, ['1J-1+0+0'], name='trojan')
    report = sim.dry_run('static')
    row = report.rows.iloc[0]
    assert row.cutoff_pred_yr == 1.0 and row.cutoff_binding == 'user' and row.expected_reason == 'undersampled'
    assert row.resample_step_yr == 0.25
    assert report.run['step_auto_yr'] == 0.25 and report.run['binding_rule'] == 'filter'


def test_auto_step_applied_before_the_run_and_numeric_output_step():
    sim = simulation(integration_years=20000, output_step='auto')
    sim.add_body(UQ, ['7E+2M-9+0+0+0'], name='1999 UQ')
    assert sim.config.Nout == 1256  # untouched until the dry-run
    report = dr.prepare_run(sim)
    # 1999 UQ: P = 1.0940602^1.5 = 1.1444 yr -> dt < 0.572 yr, snapped to 0.5 yr; Nout = 20000 / 0.5 + 1
    assert report is not None and sim.config.output_step == 0.5 and sim.config.Nout == 40001
    assert sim.config.output_step_yr == pytest.approx(0.5, rel=1e-5)
    assert dr.prepare_run(sim) is None  # once per simulation

    numeric = simulation(integration_years=20000, output_step=5)
    assert numeric.config.Nout == 4001 and numeric.config.output_step_yr == pytest.approx(5.0, rel=1e-5)
    numeric.config.tmax = int(10000 * 2 * math.pi)  # the step survives a new span
    assert numeric.config.Nout == 2001
    assert simulation(integration_years=20000, output_step=5, Nout=4000).config.Nout == 4000  # Nout overrides
    with pytest.raises(ValueError):
        simulation(output_step=-1)


def test_auto_without_mmr_keeps_the_default_nout():
    sim = simulation(integration_years=20000, output_step='auto')
    sim.add_body(HILDA, ['g-g6'], name='153')
    report = sim.dry_run('static')
    assert report.run['step_auto_yr'] is None and report.rows.empty
    assert any('MMRs only' in w for w in report.warnings)
    assert report.apply(step=True, integrator=False) == {} and sim.config.Nout == 1256


def test_report_files_and_summary(tmp_path):
    report = nea_run(4000).dry_run('static')
    files = report.to_csv(str(tmp_path / 'dry'))
    run = json.loads(open(files['dryrun_run.json']).read())
    assert run['binding_rule'] == 'short_period' and run['level'] == 'static'
    assert len(open(files['dryrun_rows.csv']).read().splitlines()) == 3
    text = report.summary(echo=False)
    assert 'aliased_unambiguous' in text and 'auto 0.5 yr' in text
    assert report.run['rows'] == 4000 and report.run['disk_GB'] > 0 and report.run['cpu_hours'] > 0


def test_formulas_against_independent_values():
    assert dr.tau_peri_yr(1.0, 0.0) == pytest.approx(1.0)
    assert dr.tau_peri_yr(4.0, 0.5) == pytest.approx(8 * math.sqrt(0.125 / 1.5))
    for alpha in (0.1, 0.5, 0.9):
        quad = integrate.quad(lambda psi, x=alpha: math.cos(psi) / (1 - 2 * x * math.cos(psi) + x**2) ** 1.5, 0, 2 * math.pi)[0] / math.pi
        assert dr.laplace_coefficient(1.5, 1, alpha) == pytest.approx(quad, rel=1e-9)
    # 21514: the prototype's Laplace-Lagrange 51.3 arcsec/yr (measured 53.5); the 0.9 alpha clip there changes little
    assert dr.laplace_lagrange_precession(2.6408075) / dr.ARCSEC == pytest.approx(51.3, abs=0.5)
    circle = dict(e=0.0, inc=0.0, Omega=0.0, omega=0.0)
    assert dr.moid_au({**circle, 'a': 1.0}, {**circle, 'a': 1.5}) == pytest.approx(0.5)  # coplanar circles
    assert dr.snap_step(3.75, (1.0, 2.0, 2.5, 5.0)) == 2.5
    assert dr.snap_step(0.03, (0.05, 0.1)) == 0.03
    assert nout_for_step(20000.0, 5.0) == 4001
