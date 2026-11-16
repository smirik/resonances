"""Pilot dry-run: a short fine integration per body (a few seconds each).

Independent references:
- 1999 UQ 7E+2M-9 (ias15): the author's 1-yr rerun (AUTHOR-FEEDBACK item 10) has 3999 turns in 20 kyr, i.e. a rate
  of 71.98 deg/yr; its 97-yr wave of 9.57 deg is the Earth 8:7 near-commensurability 7 n_E - 8 n (SAMPLING.md §5.3,
  measured on the 1-yr record and a 0.0127-yr pilot).
- 153 Hilda 3J-2: libration period 268 yr in the benchmark's 100-kyr, 1-yr record (benchmark/data/v2 summary.csv,
  geo_period_cycle_yr).
- The leak of the 15.92-yr default step (inv-09, the 1-yr benchmark records decimated by 16 and filtered at the
  physical cutoff): 0.13-0.30 deg in sigma of the Jupiter Trojans, 0.72 deg in 153 Hilda, 12.5 deg and 190 % of the
  a range in 62243 1M-4J-1, the one record the coarse step damages (non_resonant/libration -> transient/low).
Hilda and 1999 UQ need the cached solar system of 2026-09-14, 62243 that of 2026-06-09 (its benchmark epoch;
cache/solar-<timestamp>.bin); without it the test is skipped rather than downloading from Horizons.
"""

import datetime
import math
import os

import pytest

import resonances
from resonances.resonance.analysis.filtering import butterworth_gain
from resonances.simulation import dry_run as dr

UQ = dict(a=1.0940602, e=0.0155969, inc=0.4331147985225309, Omega=3.687738465129981, omega=0.13104333449701386, M=6.077459508522661)
HILDA = dict(
    a=3.9683768284310585,
    e=0.1384847450994033,
    inc=0.1366838033702228,
    Omega=3.980744967033935,
    omega=0.6818375012739626,
    M=2.4215746014696986,
)
DATE = '2026-09-14'
# 62243 at the benchmark epoch JD 2461200.5 (AstDyS osculating, benchmark/data/examples/mmr3__62243__1M-4J-1+0+0+4)
B62243 = dict(
    a=2.9755554841534546,
    e=0.0866161760010098,
    inc=0.1827676352921517,
    Omega=2.8312293225138583,
    omega=2.6098091092606692,
    M=1.0799210775949513,
)


def _solar(day: datetime.datetime) -> str:
    return os.path.join(os.getcwd(), 'cache', f'solar-{int(day.timestamp())}.bin')


SOLAR = _solar(datetime.datetime(2026, 9, 14))
pytestmark = pytest.mark.skipif(not os.path.exists(SOLAR), reason='needs the cached solar system of 2026-09-14')


def simulation(date=DATE, **kwargs):
    return resonances.Simulation(name='dryrun-pilot', date=date, save=None, plot=None, save_summary=False, **kwargs)


@pytest.mark.benchmark
def test_pilot_1999_uq_rate_forced_wave_and_ghost():
    sim = simulation(integration_years=20000, Nout=4000, integrator='ias15')
    sim.add_body(UQ, ['7E+2M-9+0+0+0'], name='1999 UQ')
    report = sim.dry_run('pilot')
    row, body = report.rows.iloc[0], report.bodies.iloc[0]
    assert row.source == 'pilot' and body['source'] == 'pilot'
    assert row.rate_deg_yr * 20000 / 360 == pytest.approx(3999, abs=2)
    # SAMPLING.md §3.1: 0.99995. The pilot is 10 P_c = 750 yr long: the slope of a 9.57-deg, 97-yr wave over L = 750 yr
    # errs by up to ~2 A / L = 4.5e-4 rad/yr, i.e. 2 x 9.57 / 750 x 5.0012 / 360 = 3.5e-4 turns per step
    assert row.turns_per_step == pytest.approx(0.99995, abs=4e-4)
    assert row.verdict == 'aliased_unambiguous' and row.nearest_ghost_au < 2e-5  # <a> sits on the ghost (4e-6 au)
    # the forced 97-yr wave passes the cutoff of 100 yr with gain G(97 yr) ~ 0.47
    period, amplitude = [float(x) for x in row.forced_terms.split(' (')[0].replace(' yr', '').replace(' deg', '').split()]
    assert '7n_Earth-8n' in row.forced_terms
    assert period == pytest.approx(97.1, abs=10)  # frequency resolution of a 1-kyr pilot: P^2 / L ~ 9 yr
    assert amplitude / butterworth_gain(97.1, row.cutoff_pred_yr, 2) == pytest.approx(9.57, abs=1.5)
    assert row.leak_slips == 0 and row.leak_deg < 1.0
    assert body['pilot_wall_s'] < 10


@pytest.mark.benchmark
def test_pilot_hilda_libration_line_and_cost():
    sim = simulation(integration_years=100000)
    sim.add_body(HILDA, ['3J-2+0-1'], name='153')
    report = sim.dry_run('pilot')
    row, body = report.rows.iloc[0], report.bodies.iloc[0]
    assert body['pilot_years'] == pytest.approx(750, abs=1)  # 10 P_c = 10 x 150 / 2 yr
    lines = [part.split(' yr')[0] for part in row.forced_terms.split('; ')]
    periods = [float(p) for p in lines]
    assert any(abs(p - 268) < 268**2 / 750 for p in periods)  # resolution P^2 / L = 96 yr
    assert row.verdict == 'resolved' and math.isfinite(row.leak_deg)
    assert body['pilot_wall_s'] < 10 and report.run['cpu_hours'] > 0
    # the default 15.92-yr step is above P_orb / 2 = 3.95 yr; the pilot accepts it: the sigma leak (inv-09: 0.72 deg)
    # is far below half the TP prominence, the a_dot error below sqrt(0.1 / 0.9) = 1/3
    assert row.expected_flags == 'short_period_aliased'
    assert row.leak_deg == pytest.approx(0.7, abs=0.3) and 2 * row.leak_deg <= row.leak_tp_prominence_deg
    assert row.leak_adot_rel < 1 / 3 and bool(row.pilot_step_ok)
    assert any('the pilot accepts the step 15.92 yr' in w for w in report.warnings)
    assert not any('short_period limit' in w for w in report.warnings)


@pytest.mark.benchmark
def test_the_pilot_step_divides_the_planned_step():
    # 15.918509 yr = 319 x 0.049901 yr: the decimated pilot has the planned cadence, not 318 x 0.05 = 15.9 yr
    fine = dr.pilot_step(15.918509, 0.05)
    assert fine <= 0.05 and 15.918509 / fine == pytest.approx(319, abs=1e-9)
    assert dr.pilot_step(0.02, 0.05) == 0.05 and dr.pilot_step(float('nan'), 0.05) == 0.05


@pytest.mark.benchmark
@pytest.mark.skipif(not os.path.exists(_solar(datetime.datetime(2026, 6, 9))), reason='needs the cached solar system of 2026-06-09')
def test_pilot_rejects_the_default_step_for_62243():
    # 62243 1M-4J-1 at 15.92 yr: the folded short-period terms reach sigma_f by degrees (inv-09: 12.5 deg over 20 kyr)
    # and a_dot by several times its own wave (190 % of the a range): the pilot does not accept the step
    sim = simulation(date='2026-06-09', integration_years=100000)
    sim.add_body(B62243, ['1M-4J-1+0+0+4'], name='62243')
    report = sim.dry_run('pilot')
    row = report.rows.iloc[0]
    assert row.expected_flags == 'short_period_aliased'
    assert row.leak_adot_rel > 1 and row.leak_deg > 1.0 and not row.pilot_step_ok
    assert any('the pilot does not accept the step 15.92 yr' in w for w in report.warnings)
    assert any('the short_period limit' in w for w in report.warnings)
