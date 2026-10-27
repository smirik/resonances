"""Secular and Lidov-Kozai verdicts of the shared analysis on signals whose answer is known by construction.

Rewrites of the TTO tests (tests/resonances/resonance/test_classify.py) and of the secular centre check
(test_secular_centre.py) to the new API (`analyse_secular`, `analyse_zlk`), with their expected verdicts:
libration -> resonant, circulation -> non_resonant, unphysical -> invalid, a bounded angle that turns back fewer
than LIBRATION_MIN_TURNING_POINTS times -> uncertain (baseline_needed). The time scales are those of real secular
(Myr) and ZLK (10 kyr) angles: the TTO tests ran 10-yr 'librations' on 100-yr records, which the new filter (P_c >=
2 x the great inequality, 1874 yr) cannot see — W4-TEST-REWRITES.md.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from resonances.resonance.analysis.secular import LongPeriodSeries, analyse_secular
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, Reason, ResonanceType, Status
from resonances.resonance.analysis.zlk import analyse_zlk
from resonances.resonance.classify import check_chaos

SETTINGS = settings_from_config()
NO_GATE = SimpleNamespace(free_gate_enabled=False, free_gate_min_cycles=3.0)
T_SECULAR = np.linspace(0.0, 1e7, 20001)  # 10 Myr at 500 yr, the benchmark's secular records
T_ZLK = np.linspace(0.0, 1e5, 10001)  # 100 kyr at 10 yr


def series(resonance_type, t, angle, e=0.1, a=2.5):
    n = len(t)
    return LongPeriodSeries(
        resonance_type, 'test', 'synthetic', t, np.asarray(angle, dtype=float), np.full(n, a), np.full(n, e) * np.ones(n), np.full(n, 0.1)
    )


def secular(angle, **kwargs):
    return analyse_secular(series(ResonanceType.SECULAR, T_SECULAR, angle, **kwargs), SETTINGS)


def zlk(angle, t=T_ZLK, **kwargs):
    return analyse_zlk(series(ResonanceType.LIDOV_KOZAI, t, angle, **kwargs), SETTINGS, gate_config=NO_GATE)


# --- test_classify.py: the verdicts --------------------------------------------------------------------------


def test_a_zlk_libration_is_resonant_with_its_period_and_centre():
    # pi + 0.5 sin(2 pi t / 10 kyr) over 100 kyr: ten cycles about pi
    analysis = zlk(np.pi + 0.5 * np.sin(2 * np.pi * T_ZLK / 1e4))
    verdict = analysis.verdict
    assert (verdict.status, verdict.angle_status) == (Status.RESONANT, AngleStatus.LIBRATION)
    assert analysis.libration_period_yr == pytest.approx(1e4, rel=0.05)
    assert analysis.numbers_row()['ep_centre_mod_2pi_deg'] == pytest.approx(180.0, abs=6.0)


def test_a_zlk_circulation_is_non_resonant_without_a_period():
    analysis = zlk(2 * np.pi * T_ZLK / 5e3)  # 20 turns
    assert (analysis.verdict.status, analysis.verdict.angle_status) == (Status.NON_RESONANT, AngleStatus.CIRCULATION)
    assert not np.isfinite(analysis.libration_period_yr)


def test_a_secular_libration_is_resonant():
    analysis = secular(np.pi + 0.5 * np.sin(2 * np.pi * T_SECULAR / 1e6))
    assert analysis.verdict.status == Status.RESONANT
    assert analysis.libration_period_yr == pytest.approx(1e6, rel=0.05)


@pytest.mark.parametrize('centre', [0.05, 1.5 * np.pi])
def test_the_centre_of_a_libration_is_where_it_is(centre):
    # TTO: the libration centre near zero (wrapped) and at 3pi/2
    analysis = zlk(centre + 0.4 * np.sin(2 * np.pi * T_ZLK / 1e4))
    measured = np.radians(analysis.numbers_row()['ep_centre_mod_2pi_deg'])
    assert abs(np.angle(np.exp(1j * (measured - centre)))) < 0.1


def test_an_unphysical_orbit_is_invalid_and_measures_nothing():
    analysis = zlk(np.pi + 0.5 * np.sin(2 * np.pi * T_ZLK / 1e4), e=1.5)
    verdict = analysis.verdict
    assert (verdict.status, verdict.reason, verdict.status_code) == (Status.INVALID, Reason.UNPHYSICAL, -99)
    assert analysis.geometry is None and not np.isfinite(analysis.libration_period_yr)
    assert analysis.numbers_row()['ctx_unphysical'] == 'hyperbolic'


def test_the_verdict_and_its_numbers_flatten_to_csv_columns():
    row = zlk(np.pi + 0.5 * np.sin(2 * np.pi * T_ZLK / 1e4)).summary_row()
    for column in ('status', 'angle_status', 'confidence', 'reason', 'status_code', 'decision_path', 'check_zlk_free_omega'):
        assert column in row
    assert row['status'] == 'resonant' and row['check_zlk_free_omega'] == 'not_measured'  # the gate is off here
    assert row['geo_period_cycle_yr'] == pytest.approx(1e4, rel=0.05)
    assert not any(column.startswith(('stall_', 'jump_')) for column in row)  # MMR-only geometry stays out


def test_an_mmr_has_no_long_period_analysis():
    with pytest.raises(KeyError):
        analyse_secular(series(ResonanceType.MMR, T_SECULAR, np.zeros_like(T_SECULAR)), SETTINGS)


# --- check_chaos (kept from test_classify.py) ---------------------------------------------------------------


def _orbit(axis, ecc):
    return SimpleNamespace(axis=np.asarray(axis, dtype=float), ecc=np.asarray(ecc, dtype=float))


@pytest.mark.parametrize(
    'axis, ecc, flag, text',
    [
        ([2.5, 2.6, 2.55], [0.1, 0.12, 0.11], 0, ''),
        ([2.5, 2.6, 2.55], [0.1, 1.5, 0.3], 1, 'hyperbolic'),
        ([2.5, -1.0, 2.55], [0.1, 0.2, 0.1], 1, 'negative_sma'),
        ([2.5, 2.6, 6.0], [0.1, 0.1, 0.1], -1, 'da_final'),  # |da|/a0 = 1.4
        ([2.5, 5.5, 2.6], [0.1, 0.1, 0.1], -1, 'da_max'),  # 1.2 at the maximum, 4% at the end
        ([0.0, 2.5, 2.5], [0.1, 0.1, 0.1], 1, 'a0=0'),
        ([2.5, 2.5, 4.975], [0.1, 0.1, 0.1], 0, ''),  # 99 %: not yet
    ],
)
def test_check_chaos(axis, ecc, flag, text):
    got, comment = check_chaos(_orbit(axis, ecc))
    assert got == flag and text in comment


# --- test_secular_centre.py: a moving centre needs a cycle and a half -------------------------------------------


def test_a_drifting_centre_is_uncertain():
    # 110 deg in 10 Myr with 270-kyr wiggles of 30 deg: bounded, never turns back
    analysis = secular(np.radians(-110.0 * T_SECULAR / 1e7 + 30.0 * np.sin(2 * np.pi * T_SECULAR / 2.7e5)))
    assert (analysis.verdict.status, analysis.verdict.reason) == (Status.UNCERTAIN, Reason.BASELINE_NEEDED)
    assert analysis.verdict.angle_status in (AngleStatus.DRIFT, AngleStatus.INSUFFICIENT_TIME)


def test_a_slow_libration_with_five_turning_points_is_resonant():
    # period 4 Myr, 80 deg: extrema at 1, 3, 5, 7, 9 Myr
    analysis = secular(np.radians(80.0 * np.sin(2 * np.pi * T_SECULAR / 4e6)))
    assert analysis.verdict.status == Status.RESONANT
    assert len(analysis.geometry.turning_points) == 5


def test_a_cycle_and_a_half_is_the_least_a_libration_needs():
    # period 6.67 Myr: extrema at 1.67, 5.0, 8.33 Myr (1020 Arcadia in z1: 1.3, 4.6, 8.0 Myr)
    analysis = secular(np.radians(80.0 * np.sin(2 * np.pi * T_SECULAR / (1e7 / 1.5))))
    assert analysis.verdict.status == Status.RESONANT
    assert len(analysis.geometry.turning_points) == SETTINGS.libration_min_turning_points == 3
    assert 'three_tp_episode' in analysis.verdict.caps


def test_one_cycle_is_too_few():
    # 80 cos(2 pi t / 8 Myr): extrema at 4 and 8 Myr inside the record (3701 Purkyne in z1, period ~8 Myr)
    analysis = secular(np.radians(80.0 * np.cos(2 * np.pi * T_SECULAR / 8e6)))
    assert (analysis.verdict.status, analysis.verdict.reason) == (Status.UNCERTAIN, Reason.BASELINE_NEEDED)
    assert len(analysis.geometry.turning_points) == 2


def test_a_single_u_is_uncertain():
    # 2005 GQ107-like: the angle rises 100 deg and comes back once, with 300-kyr wiggles of 30 deg
    analysis = secular(np.radians(100.0 * np.sin(np.pi * T_SECULAR / 1e7) + 30.0 * np.sin(2 * np.pi * T_SECULAR / 3e5)))
    assert analysis.verdict.status == Status.UNCERTAIN


def test_a_fast_libration_keeps_its_status():
    # Tina-like: 600 kyr, 35 deg about 180 deg
    analysis = secular(np.radians(180.0 + 35.0 * np.sin(2 * np.pi * T_SECULAR / 6e5)))
    assert analysis.verdict.status == Status.RESONANT
    assert analysis.libration_period_yr == pytest.approx(6e5, rel=0.03)
