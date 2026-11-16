"""The dry-run's advisory secular and ZLK rows (docs/dry-run.md): the baseline the analysis needs to confirm a libration.

Expected numbers by hand from the .env.dist defaults (read here from the file, not from the settings class):
secular P_c = max(SECULAR_FILTER_GI_FACTOR x 937 yr, SECULAR_LIBRATION_PERIOD_MIN_YR / FILTER_MIN_LIBRATION_CUTOFFS),
P_floor = max(SECULAR_LIBRATION_PERIOD_MIN_YR, FILTER_MIN_LIBRATION_CUTOFFS x P_c), N = (LIBRATION_MIN_TURNING_POINTS / 2)
P_floor + 2 FILTER_EDGE_TRIM_CUTOFFS P_c; the ZLK quadrupole time from the published planetary constants (masses
1/1047.35, 1/3497.9, 1/22902, 1/19412 of the Sun, periods 11.862, 29.457, 84.02, 164.8 yr, eccentricities 0.0484,
0.0557, 0.0472, 0.0087), not from `const`.
"""

import importlib
import math
from pathlib import Path

import pytest
from dotenv import dotenv_values

import resonances

HILDA = dict(
    a=3.9683768284310585,
    e=0.1384847450994033,
    inc=0.1366838033702228,
    Omega=3.980744967033935,
    omega=0.6818375012739626,
    M=2.4215746014696986,
)
ENV = dotenv_values(Path(importlib.import_module('resonances').__file__).parent / '.env.dist')
GREAT_INEQUALITY_YR = 937.0


def setting(name):
    return float(ENV[name])


def hand_baseline(period_min_yr, gi_factor):
    cutoff = max(gi_factor * GREAT_INEQUALITY_YR, period_min_yr / setting('FILTER_MIN_LIBRATION_CUTOFFS'))
    floor = max(period_min_yr, setting('FILTER_MIN_LIBRATION_CUTOFFS') * cutoff)
    need = setting('LIBRATION_MIN_TURNING_POINTS') / 2 * floor + 2 * setting('FILTER_EDGE_TRIM_CUTOFFS') * cutoff
    return cutoff, floor, need


def run(years, resonance, **kwargs):
    sim = resonances.Simulation(
        name='dryrun-long-period', date='2026-09-14', save=None, plot=None, save_summary=False, integration_years=years, **kwargs
    )
    sim.add_body(HILDA, [resonance], name='153')
    return sim


def test_a_100_kyr_secular_run_warns_that_500_kyr_are_needed(tmp_path):
    cutoff, floor, need = hand_baseline(setting('SECULAR_LIBRATION_PERIOD_MIN_YR'), setting('SECULAR_FILTER_GI_FACTOR'))
    assert (cutoff, floor, need) == (1e5, 2e5, 5e5)  # the defaults: 1.5 x 200 kyr + 2 x 100 kyr
    sim = run(100000, 'g-g6')
    nout = sim.config.Nout
    report = sim.dry_run('static')
    row = report.long_period.set_index('resonance').loc['g-g6']
    assert row['type'] == 'secular' and row['cutoff_pred_yr'] == pytest.approx(cutoff)
    assert row['libration_floor_yr'] == pytest.approx(floor) and row['baseline_min_yr'] == pytest.approx(need)
    assert not row['baseline_ok'] and row['step_ok']
    assert any(w.startswith('153 g-g6: span 100000 yr < 500000 yr') for w in report.warnings)
    # advisory: no MMR row, no auto step, the run's settings untouched
    assert report.rows.empty and report.run['n_long_period_rows'] == 1 and sim.config.Nout == nout
    files = report.to_csv(str(tmp_path / 'dry'))
    assert len(open(files['dryrun_long_period.csv']).read().splitlines()) == 2
    assert '1 secular/ZLK rows' in report.summary(echo=False)


def test_a_long_enough_secular_run_does_not_warn():
    report = run(500001, 'g-g6').dry_run('static')
    assert report.long_period.iloc[0]['baseline_ok']
    assert not any('g-g6' in w for w in report.warnings)


def test_an_8_kyr_zlk_run_warns_for_the_filter_and_the_gate():
    cutoff, floor, need = hand_baseline(setting('ZLK_LIBRATION_PERIOD_MIN_YR'), setting('ZLK_FILTER_GI_FACTOR'))
    assert need == pytest.approx(9370.0)  # 1.5 x 3748 + 2 x 1874
    # the quadrupole time at 153 Hilda (P = a^1.5): every giant planet lies outside its aphelion 4.52 au
    period = HILDA['a'] ** 1.5
    planets = ((1047.35, 11.862, 0.0484), (3497.9, 29.457, 0.0557), (22902.0, 84.02, 0.0472), (19412.0, 164.8, 0.0087))
    tau = 1 / sum(1 / (2 / (3 * math.pi) * m * p_j**2 / period * (1 - e_j**2) ** 1.5) for m, p_j, e_j in planets)
    report = run(8000, 'LK').dry_run('static')
    row = report.long_period.iloc[0]
    assert row['type'] == 'lidov_kozai' and row['baseline_min_yr'] == pytest.approx(need)
    assert row['zlk_time_yr'] == pytest.approx(tau, rel=0.01)
    assert row['gate_min_yr'] == pytest.approx(setting('FREE_GATE_MIN_CYCLES') * tau, rel=0.01)
    assert row['required_yr'] == pytest.approx(max(need, row['gate_min_yr']))
    warnings = [w for w in report.warnings if w.startswith('153 LK')]
    assert any('< 9370 yr' in w for w in warnings) and any('short_window' in w for w in warnings)


def test_without_the_gate_the_zlk_row_needs_only_the_filter_baseline():
    report = run(10000, 'LK', free_gate_enabled=False).dry_run('static')
    row = report.long_period.iloc[0]
    assert math.isnan(row['gate_min_yr']) and row['required_yr'] == pytest.approx(9370.0) and row['baseline_ok']


def test_a_step_above_half_the_cutoff_is_reported():
    # ZLK P_c = 1874 yr: a 1000-yr step is above P_c / 2 = 937 yr, the filter cannot be built
    report = run(100000, 'LK', Nout=101).dry_run('static')
    row = report.long_period.iloc[0]
    assert row['step_yr'] == pytest.approx(1000.0, rel=1e-3) and not row['step_ok']
    assert any('unusable_series' in w for w in report.warnings)
